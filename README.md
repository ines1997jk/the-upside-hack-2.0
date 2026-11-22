# Mentora — plateforme d'étude avec agents IA

Ce projet est un MVP web qui branche le moteur pédagogique de `main.py` dans une interface React + FastAPI.

## Ce qui est déjà intégré
- Création de matières séparées et stockage de documents dans `Cours/`, `TD/`, `Diapositives/`, `Autres/`.
- Upload de PDF, PPT/PPTX et documents.
- Détection de chapitres, planning issu du PDF et introduction de chapitre via les fonctions de ton `main.py`.
- Chatbot RAG `poser_question_au_tuteur()` du fichier original.
- Sélection/génération d'exercice avec `proposer_exercices_apres_etude()`.
- Évaluation via `evaluer_etudiant()` : indices avant la tentative 4, correction après la quatrième.
- Interface pour progression, XP, documents, planning et classement démo.

## Démarrer sous Windows
### 1) Backend
```bat
cd backend
py -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
ollama pull gemma3:1b
ollama pull nomic-embed-text
ollama serve
```
Dans un autre terminal, toujours dans `backend` :
```bat
uvicorn app.main:app --reload
```
### 2) Frontend
Dans un autre terminal :
```bat
cd frontend
npm install
npm run dev
```
Ouvre http://localhost:5173

## Important
- Le modèle configuré dans ton fichier reste `gemma3:1b`; les embeddings utilisent `nomic-embed-text`.
- Le moteur original utilise une progression JSON et certaines variables globales. Le backend les encapsule par matière pour éviter de mélanger les PDFs.
- Les slides PPT/PPTX sont stockées dans le bon dossier, mais ton `main.py` actuel analyse le PDF du cours. Pour les exploiter comme texte d'étude, il faudra ajouter l'extraction PPT/PPTX.
- Authentification, invitations réelles, synchronisation entre utilisateurs, XP persistants multi-comptes, pénalité fiable de départ (-10), suivi de présence mensuel (+30) et propositions de projets par niveau restent à connecter à une base de données et à des comptes utilisateurs. Le classement présenté est une maquette de démonstration.
- Le code n'a pas été exécuté contre Ollama ni testé avec des PDF réels dans cet environnement. Vérifie les dépendances et la qualité d'extraction sur tes documents.
