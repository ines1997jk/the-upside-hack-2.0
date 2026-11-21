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
       - SI la note est < 20 ET que tentative < 3 : Ne donne PAS la réponse. Donne un INDICE clair pour aider l'étudiant à trouver la réponse par lui-même.
       - SI la note est < 20 ET que tentative >= 3 : C'est la 3ème fois (ou plus). Donne la réponse correcte et détaillée.
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