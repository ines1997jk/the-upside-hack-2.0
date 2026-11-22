# Backend Mentora
## Installation (Windows)
```bat
cd backend
py -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```
Installe Ollama puis lance :
```bat
ollama pull gemma3:1b
ollama pull nomic-embed-text
ollama serve
```
Dans un autre terminal :
```bat
uvicorn app.main:app --reload
```
API: http://localhost:8000/docs
Le fichier `agent/main.py` est ton fichier fourni, conservé comme moteur pédagogique.
