import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialisation du modèle (llama3)
llm = ChatOllama(model="", temperature=0.3)


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


def generer_planning(retriever, duree_jours=7):
    # On demande à l'IA de créer un planning structuré en JSON
    prompt = f"""
    Tu es un agent planificateur. En te basant sur le cours fourni, crée un planning d'étude sur {duree_jours} jours.
    L'étudiant doit réviser une petite partie par jour.
    Renvoie UNIQUEMENT un objet JSON valide avec cette structure :
    [
      {{"jour": 1, "titre": "Titre du chapitre", "objectif": "Ce qu'il doit savoir faire"}},
      {{"jour": 2, "titre": "...", "objectif": "..."}}
    ]
    Voici le contenu du cours : {retriever.invoke("syllabus chapitres")[0].page_content[:2000]}
    """

    response = llm.invoke(prompt)
    return response.content

# ---------------------------------------------------------
# PROBLÈME 3 : L'ÉVALUATEUR & LACUNES
# ---------------------------------------------------------


def evaluer_etudiant(reponse_etudiant, sujet):
    prompt = f"""
    Tu es un professeur qui évalue un étudiant sur le sujet : {sujet}.
    La réponse de l'étudiant est : "{reponse_etudiant}"
    
    Ton rôle :
    1. Donne une note sur 20.
    2. Identifie la LACUNE (pourquoi il n'a pas eu 20).
    3. Donne un feedback motivant sur ce qu'il a bien réussi (Points forts).
    Sois bref et pédagogique.
    """
    response = llm.invoke(prompt)
    return response.content


# ==========================================
# ZONE DE TEST (C'est ici qu'on va exécuter)
# ==========================================
if __name__ == "__main__":
    # --- REMPLACEZ "mon_cours.pdf" PAR UN VRAI PDF DANS VOTRE DOSSIER ---
    # (Vous pouvez télécharger un cours de maths ou d'informatique au hasard sur internet pour tester)

    fichier_test = "mon_cours.pdf"

    # 1. On charge le PDF dans la mémoire de l'IA
    if os.path.exists(fichier_test):
        memoire = charger_et_creer_memoire(fichier_test)
        print("✅ Mémoire de l'agent initialisée !\n")

        # 2. Test du Tuteur (Problème 1)
        print("--- 🤖 TEST DU TUTEUR ---")
        reponse_tuteur = poser_question_au_tuteur(
            memoire, "Je n'ai pas compris le concept principal de ce cours. Explique-le moi.")
        print(reponse_tuteur)
        print("\n")

        # 3. Test du Planificateur (Problème 2)
        print("--- 📅 TEST DU PLANIFICATEUR ---")
        planning = generer_planning(memoire, duree_jours=5)
        print(planning)
        print("\n")

        # 4. Test de l'Évaluateur (Problème 3)
        print("--- 📝 TEST DE L'ÉVALUATEUR ---")
        feedback = evaluer_etudiant(
            "Je pense que c'est un système qui permet de faire circuler l'énergie", "Les bases du cours")
        print(feedback)

    else:
        print(f" Erreur : Le fichier {fichier_test} est introuvable. Mettez un PDF dans le dossier pour tester.")
