import json
import os
import re
import unicodedata
from datetime import date
from pypdf import PdfReader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialisation du modèle (llama3)
llm = ChatOllama(model="gemma3:1b", temperature=0.3)


def charger_et_creer_memoire(fichier_pdf):
    print(f"Lecture du fichier {fichier_pdf}...")
    loader = PyPDFLoader(fichier_pdf)
    docs = loader.load()

    # Découper le texte en petits morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Créer la base de données vectorielle (Mémoire de l'agent)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectorstore.as_retriever()
    return retriever

# ---------------------------------------------------------
# PROBLÈME 1 : LE TUTEUR (Compréhension)
# ---------------------------------------------------------


def poser_question_au_tuteur(retriever, question_etudiant):
    # Le prompt force l'IA à suivre votre méthode pédagogique
    system_prompt = (
        "Tu es un tuteur pédagogique expert. Tu dois répondre à la question de l'étudiant "
        "UNIQUEMENT en te basant sur le contexte fourni (son cours).\n"
        "Structure ta réponse EXACTEMENT comme ceci :\n"
        "1. 🎨 **L'Analogie / L'Explication Visuelle** : Explique le concept comme si l'étudiant avait 10 ans, avec une image mentale.\n"
        "2. 🧪 **L'Exemple Concret** : Donne un exemple d'application.\n"
        "3. 📖 **La Définition Formelle** : Donne la définition exacte du cours.\n"
        "4. 💡 **Indice pour progresser** : Ne donne pas la réponse finale s'il s'agit d'un exercice, donne un indice.\n\n"
        "Contexte du cours :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question_etudiant})
    return response["answer"]

# ---------------------------------------------------------
# PROBLÈME 2 : LE PLANIFICATEUR (Agent)
# ---------------------------------------------------------

def _parser_json_robuste(contenu):
    """Extrait un objet JSON valide même si le modèle ajoute du texte."""
    if not contenu:
        return None
    contenu = contenu.strip()
    contenu = re.sub(r"^```(?:json)?\s*", "", contenu, flags=re.IGNORECASE)
    contenu = re.sub(r"\s*```$", "", contenu).strip()

    try:
        return json.loads(contenu)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", contenu, re.DOTALL)
    if not match:
        return None
    bloc = match.group(0)
    bloc = bloc.replace("\u201c", '"').replace("\u201d", '"')
    bloc = bloc.replace("\u2018", "'").replace("\u2019", "'")

    try:
        return json.loads(bloc)
    except json.JSONDecodeError:
        bloc = re.sub(r",\s*([\]}])", r"\1", bloc)
        try:
            return json.loads(bloc)
        except json.JSONDecodeError:
            return None


def _appeler_llm_json(prompt, max_tentatives=1):
    """Appelle le LLM en mode JSON avec retry + parsing robuste."""
    for _ in range(max_tentatives):
        try:
            response = llm.invoke(prompt, format="json")
        except TypeError:
            response = llm.invoke(prompt)

        data = _parser_json_robuste(response.content)
        if data is not None:
            return data

        prompt = (
            "Ta réponse précédente n'était pas un JSON valide. "
            "Renvoie UNIQUEMENT un objet JSON, sans texte autour, "
            "sans backticks et avec des guillemets droits.\n\n"
            f"Réponse précédente :\n{response.content}"
        )
    return None



def _normaliser_cle(titre):
    titre = titre.replace("'", "").replace("’", "").replace("`", "")
    titre = unicodedata.normalize("NFD", titre.lower())
    titre = "".join(
        caractere
        for caractere in titre
        if unicodedata.category(caractere) != "Mn"
    )
    return " ".join(titre.split())


def _titre_plausible(titre, texte_page):
    if not titre:
        return False

    titre = " ".join(titre.split())
    if len(titre) < 4 or len(titre) > 100:
        return False

    lettres = sum(caractere.isalpha() for caractere in titre)
    if lettres < 3:
        return False

    symboles = sum(
        not (
            caractere.isalnum()
            or caractere.isspace()
            or caractere in "'’-"
        )
        for caractere in titre
    )
    if symboles / len(titre) > 0.2:
        return False

    mots = titre.split()
    if all(len(mot) <= 2 for mot in mots):
        return False

    if titre.endswith((".", ",", ";", ":")):
        return False
    if titre.lower().startswith(
        ("soit ", "on ", "si ", "alors ", "supposons ",
         "montrons ", "pour ", "posons ")
    ):
        return False

    debut = _normaliser_cle(texte_page[:400])
    if _normaliser_cle(titre) not in debut:
        return False

    return True


def _titres_depuis_signets(fichier_pdf):
    try:
        reader = PdfReader(fichier_pdf)
        resultat = {}

        def parcourir(items):
            for item in items:
                if isinstance(item, list):
                    parcourir(item)
                else:
                    try:
                        page = reader.get_destination_page_number(item) + 1
                        titre = getattr(item, "title", "")
                        if titre:
                            resultat.setdefault(page, titre.strip())
                    except Exception:
                        continue

        parcourir(reader.outline)
        return resultat
    except Exception:
        return {}


MOTIF_CHAPITRE = re.compile(
    r"^\s*(chapitre|chapter|partie|part|module)\s+"
    r"[\dIVXivx]+(?:\b|[.\-)]).{3,80}$",
    re.IGNORECASE | re.MULTILINE,
)


def _titre_par_regles(texte_page):
    premieres_lignes = "\n".join(
        texte_page.strip().splitlines()[:6]
    )
    correspondance = MOTIF_CHAPITRE.search(premieres_lignes)
    if correspondance:
        titre = correspondance.group(0).strip()
        if _titre_plausible(titre, texte_page):
            return titre
    return None


def extraire_chapitres_et_pages(fichier_pdf):
    """
    Identifie les chapitres et associe chaque page au chapitre courant.
    """

    loader = PyPDFLoader(fichier_pdf)
    pages = loader.load()
    signets = _titres_depuis_signets(fichier_pdf)

    chapitres = {}
    cles_vues = {}
    chapitre_actuel = "Introduction du cours"

    for i, page in enumerate(pages):
        texte = page.page_content.strip()
        if not texte:
            continue

        numero = i + 1
        nouveau_titre = signets.get(numero)

        if not nouveau_titre and i > 0:
            nouveau_titre = _titre_par_regles(texte)

        if not nouveau_titre and not signets and i > 0:
            prompt = f"""
Tu analyses une page d'un cours universitaire.

PAGE {numero} :
{texte[:3000]}

Identifie uniquement le titre d'un nouveau chapitre.
N'extrais aucune formule, définition, section secondaire ou phrase.
Le titre doit apparaître clairement au début de la page.

Renvoie uniquement :
{{"nouveau_chapitre": null}}
ou
{{"nouveau_chapitre": "Titre exact du chapitre"}}
"""
            resultat = _appeler_llm_json(prompt)
            if isinstance(resultat, dict):
                candidat = resultat.get("nouveau_chapitre")
                if (
                    isinstance(candidat, str)
                    and _titre_plausible(candidat, texte)
                ):
                    nouveau_titre = candidat

        if nouveau_titre:
            nouveau_titre = " ".join(nouveau_titre.split())
            cle = _normaliser_cle(nouveau_titre)
            chapitre_actuel = cles_vues.setdefault(
                cle,
                nouveau_titre
            )

        if chapitre_actuel not in chapitres:
            chapitres[chapitre_actuel] = {
                "pages": [],
                "texte": ""
            }

        chapitres[chapitre_actuel]["pages"].append(numero)
        chapitres[chapitre_actuel]["texte"] += "\n\n" + texte

    return chapitres


import unicodedata


def generer_introduction_chapitre(
    texte_chapitre,
    chapitre_precedent,
    chapitre_actuel
):
    """
    Génère l'introduction à partir du texte exact du chapitre.
    """

    prompt = f"""
Tu es un professeur pédagogique.

Tu dois préparer l'étudiant avant qu'il commence
le chapitre suivant :

CHAPITRE ACTUEL :
{chapitre_actuel}

CHAPITRE PRÉCÉDENT :
{chapitre_precedent or "Aucun"}

CONTENU DU CHAPITRE ACTUEL :
{texte_chapitre[:12000]}

Prépare une introduction pédagogique.

1. RAPPEL :
Si un chapitre précédent existe, rappelle les notions
qui seront utiles pour comprendre le chapitre actuel.
Sinon, présente les prérequis utiles.

2. LIEN :
Explique le lien entre le chapitre précédent
et le chapitre actuel.
S'il n'existe pas de lien évident, dis-le honnêtement.

3. IMPORTANCE :
Explique pourquoi le chapitre actuel est important.

4. APPLICATIONS RÉELLES :
Donne 2 à 4 applications concrètes dans le monde réel.

5. OBJECTIF :
Explique ce que l'étudiant devrait savoir faire
après avoir étudié ce chapitre.

Règles :
- Le sujet principal doit être le chapitre actuel.
- Utilise le contenu fourni pour comprendre le chapitre.
- Ne confonds pas le chapitre avec une section secondaire.
- N'invente pas de théorèmes ou de définitions du cours.
- Tu peux utiliser tes connaissances générales pour
  les applications réelles.
- Réponds en français simple et pédagogique.

Renvoie uniquement un JSON valide :
{{
    "rappel": "...",
    "lien": "...",
    "importance": "...",
    "applications_reelles": ["...", "..."],
    "objectif": "..."
}}
"""

    resultat = _appeler_llm_json(prompt)

    if not isinstance(resultat, dict):
        return {
            "rappel": "Introduction indisponible.",
            "lien": "",
            "importance": "",
            "applications_reelles": [],
            "objectif": ""
        }

    return resultat

def normaliser_texte(texte):
    """Enlève les accents et met le texte en minuscules."""
    texte = texte.lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(
        c for c in texte
        if unicodedata.category(c) != "Mn"
    )
    return texte


def chercher_exercices_dans_pdf(fichier_pdf, partie):
    """
    Cherche des exercices dans le PDF qui correspondent
    à la partie étudiée.
    """
    loader = PyPDFLoader(fichier_pdf)
    pages = loader.load()

    partie_normalisee = normaliser_texte(partie)

    # Mots importants du titre
    mots_partie = [
        mot for mot in re.findall(r"\b\w+\b", partie_normalisee)
        if len(mot) > 3
        and mot not in {
            "chapitre", "introduction", "fondements",
            "theorie", "notions", "cours"
        }
    ]

    exercices_trouves = []

    # "Application" désigne aussi un type de théorème dans certains cours
    # : il ne doit donc jamais compter comme marqueur d'exercice.
    pattern = (
        r"(?im)(?=^\s*(?:"
        r"exercice(?:s)?\b|"
        r"probl[eè]me(?:s)?\b|"
        r"\btd\s*\d*"
        r")[\s:.-]*\d*)"
    )

    for page in pages:
        texte = page.page_content
        texte_normalise = normaliser_texte(texte)

        contient_exercice = (
            re.search(
                r"(?im)^\s*(?:exercice(?:s)?\b|"
                r"probl[eè]me(?:s)?\b|\btd\s*\d*)",
                texte
            )
            is not None
        )

        if not contient_exercice:
            continue

        # Découpe la page en blocs d'exercices
        blocs = re.split(pattern, texte)

        for bloc in blocs:
            bloc = bloc.strip()

            if len(bloc) < 30 or len(bloc) > 5000:
                continue

            if not re.match(
                r"(?is)^\s*(?:exercice(?:s)?\b|"
                r"probl[eè]me(?:s)?\b|\btd\s*\d*)",
                bloc
            ):
                continue

            bloc_normalise = normaliser_texte(bloc)

            # Vérifie si le bloc parle de la partie étudiée
            correspondances = sum(
                1 for mot in mots_partie
                if mot in bloc_normalise
            )

            # Au moins un mot important doit correspondre
            if correspondances >= 1:
                exercices_trouves.append(bloc)

    return exercices_trouves

def proposer_exercices_apres_etude(fichier_pdf, partie, retriever=None, niveau="Facile"):
    """
    1. Cherche des exercices (TD) dans le PDF.
    2. Si aucun exercice correspondant n'est trouvé,
       génère un exercice avec le LLM (generer_exercice).
    """
    exercices_pdf = chercher_exercices_dans_pdf(
        fichier_pdf,
        partie
    )

    if exercices_pdf:
        return {
            "partie": partie,
            "source": "PDF",
            "message": "Exercices trouvés dans le cours.",
            "exercices": exercices_pdf
        }

    print("Aucun exercice correspondant trouvé dans le PDF.")
    print("Génération d'un exercice avec le LLM...")

    if retriever is None:
        retriever = obtenir_retriever(fichier_pdf)

    try:
        exercice = generer_exercice(retriever, partie, niveau)
    except Exception as erreur:
        return {
            "partie": partie,
            "source": "LLM",
            "erreur": f"Impossible de générer l'exercice : {erreur}",
            "exercices": []
        }

    return {
        "partie": partie,
        "source": "LLM",
        "message": "Aucun TD trouvé : exercice généré à partir du cours.",
        "exercices": [exercice]
    }

def initialiser_progression(fichier_pdf):
    chapitres = extraire_chapitres_et_pages(fichier_pdf)
    titres_chapitres = [
        titre for titre in chapitres
        if titre != "Introduction du cours"
    ]

    if not titres_chapitres:
        return {"erreur": "Aucun chapitre extrait du cours."}

    etat = {
        "fichier_pdf": fichier_pdf,
        "date_debut": date.today().isoformat(),
        "jour_etude": 1,
        "chapitres": chapitres,
        "titres_chapitres": titres_chapitres,
        "index_chapitre_actuel": 0,
        "index_partie_actuelle": 0,
        "parties_validees": [],
        "chapitre_precedent": None,
        "chapitre_actuel": titres_chapitres[0],
        "partie_actuelle": titres_chapitres[0],
        "statut": "en_cours"
    }

    sauvegarder_progression(etat)
    return etat


def sauvegarder_progression(etat):
    with open("progression.json", "w", encoding="utf-8") as f:
        json.dump(etat, f, indent=4, ensure_ascii=False)


def charger_progression():
    if not os.path.exists("progression.json"):
        return None

    with open("progression.json", "r", encoding="utf-8") as f:
        etat = json.load(f)

    fichier_pdf = etat.get("fichier_pdf", "mon_cours.pdf")
    titres = etat.get("titres_chapitres", [])

    def titre_sauvegarde_invalide(titre):
        if not isinstance(titre, str):
            return True
        titre = " ".join(titre.split())
        if len(titre) < 4 or len(titre) > 100:
            return True
        if titre.lower().startswith(
            ("soit ", "on ", "si ", "alors ",
             "supposons ", "montrons ", "pour ", "posons ")
        ):
            return True
        symboles = sum(
            not (caractere.isalnum() or caractere.isspace())
            for caractere in titre
        )
        return symboles / len(titre) > 0.2

    sauvegarde_invalide = (
        "chapitres" not in etat
        or any(titre_sauvegarde_invalide(titre) for titre in titres)
    )

    if sauvegarde_invalide and os.path.exists(fichier_pdf):
        nouveau_etat = initialiser_progression(fichier_pdf)
        if "erreur" not in nouveau_etat:
            return nouveau_etat

    return etat


def _est_titre_chapitre(titre):
    motif_chapitre = re.compile(
        r"^(chapitre|chapter|partie|module)\b",
        re.IGNORECASE
    )
    return bool(motif_chapitre.match(titre))


def _numero_chapitre(titre):
    correspondance = re.match(r"^(\d+)", titre)
    return correspondance.group(1) if correspondance else None


def _identifier_chapitres(parties, index):
    partie_actuelle = parties[index]
    partie_precedente = parties[index - 1] if index > 0 else None

    numero_actuel = _numero_chapitre(partie_actuelle)
    numero_precedent = (
        _numero_chapitre(partie_precedente)
        if partie_precedente else None
    )

    chapitre_actuel = partie_actuelle
    chapitre_precedent = None

    for position in range(index, -1, -1):
        titre = parties[position]
        if _est_titre_chapitre(titre):
            chapitre_actuel = titre
            break

    for position in range(index - 1, -1, -1):
        titre = parties[position]
        if _est_titre_chapitre(titre):
            chapitre_precedent = titre
            break

    nouveau_chapitre = (
        index == 0
        or (
            not _est_titre_chapitre(partie_actuelle)
            and numero_actuel is not None
            and numero_precedent is not None
            and numero_actuel != numero_precedent
        )
        or (
            _est_titre_chapitre(partie_actuelle)
            and partie_precedente != partie_actuelle
        )
    )

    return chapitre_precedent, chapitre_actuel, nouveau_chapitre


def obtenir_planning_adaptatif(etat):
    index_chapitre = etat["index_chapitre_actuel"]
    titres = etat["titres_chapitres"]

    if index_chapitre >= len(titres):
        etat["statut"] = "termine"
        sauvegarder_progression(etat)
        return {
            "statut": "termine",
            "message": "Toutes les parties sont maîtrisées."
        }

    chapitre_actuel = titres[index_chapitre]
    chapitre_precedent = (
        titres[index_chapitre - 1]
        if index_chapitre > 0
        else None
    )
    partie_actuelle = etat["partie_actuelle"]
    afficher_introduction = etat["index_partie_actuelle"] == 0

    return {
        "jour_etude": etat["jour_etude"],
        "chapitre_precedent": chapitre_precedent,
        "chapitre_actuel": chapitre_actuel,
        "partie_actuelle": partie_actuelle,
        "afficher_introduction": afficher_introduction,
        "objectif": f"Maîtriser : {partie_actuelle}",
        "parties_validees": etat["parties_validees"],
        "parties_restantes": titres[index_chapitre + 1:],
        "message": (
            "La partie suivante sera débloquée "
            "après validation avec 20/20."
        )
}


def mettre_a_jour_progression(etat, resultat_evaluation):
    required = {"score_sur_20", "lacunes", "maitrise_validee"}
    missing = required.difference(resultat_evaluation)
    if missing:
        raise ValueError(
            "Résultat d'évaluation incomplet : "
            + ", ".join(sorted(missing))
        )

    score = resultat_evaluation["score_sur_20"]
    partie = etat["partie_actuelle"]
    etat["dernier_resultat"] = resultat_evaluation

    if score == 20 and resultat_evaluation["maitrise_validee"]:
        if partie not in etat["parties_validees"]:
            etat["parties_validees"].append(partie)

        etat["index_chapitre_actuel"] += 1
        etat["index_partie_actuelle"] = 0
        etat["jour_etude"] += 1

        if etat["index_chapitre_actuel"] < len(etat["titres_chapitres"]):
            etat["chapitre_precedent"] = etat["chapitre_actuel"]
            etat["chapitre_actuel"] = etat["titres_chapitres"][
                etat["index_chapitre_actuel"]
            ]
            etat["partie_actuelle"] = etat["chapitre_actuel"]
            etat["statut"] = "en_cours"
        else:
            etat["statut"] = "termine"
    else:
        etat["statut"] = "remediation"

    sauvegarder_progression(etat)
    return obtenir_planning_adaptatif(etat)

# ---------------------------------------------------------
# PROBLÈME 3 : L'ÉVALUATEUR & LACUNES
# ---------------------------------------------------------
def extraire_note(feedback_texte):
    match = re.search(r"\[NOTE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*20\]", feedback_texte)
    if match:
        return float(match.group(1))
    return 0.0 

def generer_exercice(retriever, sujet, niveau="Facile"):
    contexte = retriever.invoke(sujet)[0].page_content
    
    prompt = f"""
    Tu es un professeur. En te basant UNIQUEMENT sur ce contexte du cours :
    {contexte}
    
    Génère un exercice évaluatif clair et précis sur le sujet : {sujet}.
    L'exercice doit tester la compréhension profonde, pas juste du par-cœur.
    Génère un exercice évaluatif de niveau **{niveau}** sur le sujet : {sujet}.
    
    Voici ce que j'attends selon le niveau :
    - **Facile** : Une question de cours directe ou une application très simple (ex: définir une notion, donner un exemple basique).
    - **Moyen** : Une application nécessitant une réflexion ou un calcul en plusieurs étapes (ex: utiliser le théorème dans un cas concret).
    - **Difficile** : Un problème complexe, une démonstration ou un cas général (ex: prouver une propriété ou analyser un contre-exemple).
    
    Ne donne PAS la réponse, donne juste la question de l'exercice.
    """
    response = llm.invoke(prompt)
    return response.content

def evaluer_etudiant(reponse_etudiant, question, sujet, tentative=1):
    prompt = f"""
    Tu es un système d'évaluation académique impitoyable et ultra-strict. 
    Ton rôle est de déterminer si un étudiant est autorisé à passer à la section suivante de son cours.
    
    RÈGLE DE NOTATION ABSOLUE (Ne déroge JAMAIS à ces règles, sois impitoyable sur la note) :
    - Si la réponse est fausse, hors sujet ou inventée : NOTE = 2/20 (Maximum).
    - Si la réponse est incomplète (il manque des mots-clés importants) : NOTE = 10/20 (Maximum).
    - Si la réponse est bonne mais qu'il manque une seule nuance : NOTE = 15/20 (Maximum).
    - Si la réponse est parfaite et exhaustive : NOTE = 20/20.
    *La note ne doit en aucun cas refléter de la sympathie ou de l'encouragement.*
    
    Sujet : {sujet}
    Question posée : {question}
    Réponse de l'étudiant : "{reponse_etudiant}"
    Numéro de tentative de l'étudiant : {tentative}
    
    Voici le BARÈME ULTRA-STRICT à appliquer :
    - 0 à 5 : Hors sujet, réponse complètement fausse, ou invention de concepts.
    - 6 à 12 : L'étudiant a compris l'idée générale mais il y a des erreurs de vocabulaire ou des approximations.
    - 13 à 19 : La réponse est très bonne, mais il manque UNE précision ou un détail important.
    - 20 : La réponse est parfaite, exhaustive, utilise le vocabulaire exact du cours et répond à tous les aspects de la question.
    
    Tu DOIS structurer ta réponse EXACTEMENT dans cet ordre, avec ces titres :
    
    1. ❌ **Les Lacunes** : Qu'est-ce qui manque ou est faux? Liste de manière objective et factuelle les éléments qui manquent ou qui sont incorrects. Ne sois pas sévère dans le ton, sois juste précis sur l'information manquante.
    2. ✅ **Les Points Forts** : Qu'a-t-il bien compris, même si c'est infime ?
    3. 🔥 **Feedback** : Sois direct. S'il a moins de 20, dis-lui qu'il doit réviser cette partie avant de continuer.
    4. 🔍 **Analyse des erreurs** : Explique-lui en détail quelles notions il a utilisées à tort, et pourquoi ce n'était pas la bonne approche.
    5. 💡 **Aide (Indice ou Réponse)** : 
       - SI la note est < 20 ET que tentative <=3 : Ne donne PAS la réponse. Donne un INDICE clair pour aider l'étudiant à trouver la réponse par lui-même.
       - SI la note est < 20 ET que tentative == 4 : C'est la 4ème fois. l'agent donne la réponse correcte et détaillée.
    6. 📊 **La Note Finale** : Donne la note sur 20 sous ce format exact : [NOTE : X/20].
    7. 🔒 **Statut de progression** : Si la note est strictement égale à 20/20, écris "[STATUT : DÉBLOQUÉ]". Sinon, écris "[STATUT : VERROUILLÉ - RÉVISION NÉCESSAIRE]".
    """
    response = llm.invoke(prompt)
    return response.content

def generer_exercice_remediation(retriever, sujet, feedback_precedent, ancien_exercice, niveau_actuel):
    contexte = retriever.invoke(sujet)[0].page_content
    
    prompt = f"""
    Tu es un professeur. L'étudiant a échoué à l'exercice de niveau {niveau_actuel}.
    Voici l'ancien exercice (INTERDIT de le redonner) : "{ancien_exercice}"
    Voici le feedback de ses erreurs : "{feedback_precedent}"
    
    En te basant UNIQUEMENT sur ce contexte du cours :
    {contexte}
    
    Ta mission : Génère un NOUVEL exercice de niveau {niveau_actuel}, mais plus guidé ou plus simple dans sa formulation, 
    qui cible SPÉCIFIQUEMENT les lacunes de l'étudiant pour l'aider à surmonter ses erreurs.
    Ne donne pas la réponse. Donne juste le nouvel exercice.
    """
    response = llm.invoke(prompt)
    return response.content


def initialiser_exercice(student_answer="", correct_answer="", attempts=0):
    return {
        "attempts": attempts,
        "hints_used": 0,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "show_solution": False
    }


def soumettre_reponse_exercice(exercise_state, reponse_etudiant, exercice):
    exercise_state["attempts"] += 1
    exercise_state["student_answer"] = reponse_etudiant

    reponse_normalisee = reponse_etudiant.strip().casefold()
    correction_normalisee = exercise_state["correct_answer"].strip().casefold()
    reponse_correcte = (
        bool(correction_normalisee)
        and reponse_normalisee == correction_normalisee
    )

    if reponse_correcte:
        return {
            "correct": True,
            "message": "Bravo, ta reponse est correcte.",
            "exercise_state": exercise_state
        }

    if exercise_state["attempts"] >= 3:
        exercise_state["show_solution"] = True

    if exercise_state["show_solution"]:
        niveau_aide = "montre la correction complete avec les etapes"
    else:
        niveau_aide = "explique l'erreur et donne seulement un indice pour continuer"

    prompt = f"""
    Tu es un tuteur pedagogique.

    Exercice : {exercice}
    Reponse de l'etudiant : {reponse_etudiant}
    Reponse attendue : {exercise_state["correct_answer"]}
    Consigne : {niveau_aide}

    Ne juge pas l'etudiant. Explique clairement l'erreur et aide-le a poursuivre.
    """
    response = llm.invoke(prompt)
    return {
        "correct": False,
        "message": response.content,
        "exercise_state": exercise_state
    }


def demander_correction(exercise_state):
    exercise_state["show_solution"] = True
    return {
        "correction": exercise_state["correct_answer"],
        "exercise_state": exercise_state
    }



# ==========================================
# 4. MODULE INDÉPENDANT : GAMIFICATION (Optionnel)
# ==========================================
classement_amis = {"Vous": 0, "Amine": 150, "Sarah": 200, "Yassine": 50}

def ajouter_points(nom_joueur, points):
    if nom_joueur in classement_amis:
        classement_amis[nom_joueur] += points
        print(f"🎁 [GAMIFICATION] +{points} XP pour {nom_joueur} !")

def afficher_classement():
    print("\n🏆 --- CLASSEMENT DU GROUPE D'AMIS --- 🏆")
    classement_trie = sorted(classement_amis.items(), key=lambda x: x[1], reverse=True)
    for position, (nom, score) in enumerate(classement_trie, 1):
        if nom == "Vous":
            print(f"👉 {position}. {nom} (C'est vous !) : {score} XP 👈")
        else:
            print(f"   {position}. {nom} : {score} XP")
    print("--------------------------------------\n")


# ---------------------------------------------------------
# LIAISON : TD du PDF (sinon exercice généré) → ÉVALUATION
#           → MISE À JOUR DU PLANNING → XP
# ---------------------------------------------------------
_cache_retrievers = {}


def obtenir_retriever(fichier_pdf):
    """Crée la mémoire vectorielle une seule fois par PDF."""
    if fichier_pdf not in _cache_retrievers:
        _cache_retrievers[fichier_pdf] = charger_et_creer_memoire(fichier_pdf)
    return _cache_retrievers[fichier_pdf]


def _extraire_lacunes(feedback_texte):
    match = re.search(
        r"Les Lacunes\**\s*:?(.*?)(?=\n\s*2\.|✅|$)",
        feedback_texte,
        re.DOTALL
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return "Voir le feedback complet."


def construire_resultat_evaluation(feedback_texte):
    """Convertit le feedback du LLM au format attendu par mettre_a_jour_progression."""
    note = extraire_note(feedback_texte)
    maitrise = note >= 20 and "VERROUILLÉ" not in feedback_texte
    return {
        "score_sur_20": note,
        "lacunes": _extraire_lacunes(feedback_texte),
        "maitrise_validee": maitrise
    }


def lancer_session_exercices(etat, niveau="Facile"):
    """
    1. Cherche un TD dans le PDF pour la partie actuelle.
    2. S'il n'y en a pas, génère l'exercice avec le LLM.
    3. Évalue la réponse, met à jour le planning à CHAQUE exercice,
       donne des XP, et génère un exercice de remédiation en cas d'échec.
    """
    partie = etat["partie_actuelle"]
    fichier_pdf = etat["fichier_pdf"]

    proposition = proposer_exercices_apres_etude(fichier_pdf, partie, niveau=niveau)
    exercices = proposition.get("exercices", [])
    if not exercices:
        print(proposition.get("erreur", "Aucun exercice disponible."))
        return obtenir_planning_adaptatif(etat)

    premier = exercices[0]
    question = premier if isinstance(premier, str) else premier.get("enonce", str(premier))

    tentative = 1
    planning = obtenir_planning_adaptatif(etat)

    while True:
        print(f"\n📝 EXERCICE (source : {proposition.get('source')}, tentative {tentative}) :")
        print(question)

        reponse = input("\n✍️ Ta réponse (ou 'stop' pour arrêter) : ").strip()
        if reponse.lower() == "stop":
            break

        feedback = evaluer_etudiant(reponse, question, partie, tentative)
        print("\n" + feedback)

        resultat = construire_resultat_evaluation(feedback)

        # Le planning s'adapte après chaque exercice
        planning = mettre_a_jour_progression(etat, resultat)

        ajouter_points("Vous", int(resultat["score_sur_20"]))

        if resultat["maitrise_validee"]:
            ajouter_points("Vous", 50)
            print("\n🔓 Partie validée, la suite est débloquée !")
            break

        print("\n🔁 Nouvel exercice de remédiation ciblé sur tes lacunes...")
        question = generer_exercice_remediation(
            obtenir_retriever(fichier_pdf),
            partie,
            feedback,
            question,
            niveau
        )
        tentative += 1

    afficher_classement()
    return planning
# ==========================================
# ZONE DE TEST
# ==========================================

if __name__ == "__main__":

    fichier_test = "mon_cours.pdf"

    if os.path.exists(fichier_test):
        etat = charger_progression()
        if etat is None:
            etat = initialiser_progression(fichier_test)

        print("\n📚 PLAN DU COURS")
        for titre, infos in etat["chapitres"].items():
            print(f"\n{titre}")
            print("Pages :", infos["pages"])

        planning = obtenir_planning_adaptatif(etat)

        if planning.get("statut") != "termine":
            chapitre_actuel = planning["chapitre_actuel"]
            texte_chapitre = etat["chapitres"][chapitre_actuel]["texte"]
            chapitre_precedent = planning["chapitre_precedent"]

            if planning["afficher_introduction"]:
                introduction = generer_introduction_chapitre(
                    texte_chapitre,
                    chapitre_precedent,
                    chapitre_actuel
                )

                print("\n📖 INTRODUCTION AU CHAPITRE")
                if chapitre_precedent:
                    print("\n🔁 Rappel :")
                    print(introduction.get("rappel", ""))

                    print("\n🔗 Lien :")
                    print(introduction.get("lien", ""))

                print("\n⭐ Importance :")
                print(introduction.get("importance", ""))

                print("\n🌍 Applications réelles :")
                applications = introduction.get(
                    "applications_reelles", []
                )

                if isinstance(applications, str):
                    applications = [applications]

                if applications:
                    for application in applications:
                        print("-", application)
                else:
                    print("- Non renseignées")

                print("\n🎯 Objectif :")
                print(introduction.get("objectif", ""))

        print("\n📅 Planning actuel :")
        print(json.dumps(planning, indent=4, ensure_ascii=False))

        if planning.get("statut") != "termine":
            planning = lancer_session_exercices(etat)
            print("\n📅 Planning mis à jour :")
            print(json.dumps(planning, indent=4, ensure_ascii=False))

    else:
        print("PDF introuvable :", fichier_test)