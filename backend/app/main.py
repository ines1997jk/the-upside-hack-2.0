import os, json, uuid, shutil, sys, traceback, hashlib, secrets
from pathlib import Path
from datetime import date, datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUBJECTS = DATA / "subjects"
SUBJECTS.mkdir(parents=True, exist_ok=True)
USERS_PATH = DATA / "users.json"
SESSIONS = {}
if not USERS_PATH.exists():
    USERS_PATH.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
sys.path.insert(0, str(ROOT / "agent"))
import main as agent
# The web API persists progression per subject, not in main.py's shared cwd file.
agent.sauvegarder_progression = lambda etat: None

app = FastAPI(title="Mentora API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def subject_dir(sid): return SUBJECTS / sid
def meta_path(sid): return subject_dir(sid) / "subject.json"
def load_meta(sid):
    p=meta_path(sid)
    if not p.exists(): raise HTTPException(404, "Matière introuvable")
    return json.loads(p.read_text(encoding="utf-8"))
def save_meta(sid, data): meta_path(sid).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
def state_path(sid): return subject_dir(sid) / "progression.json"
def load_state(sid):
    p=state_path(sid)
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))
def save_state(sid,state): state_path(sid).write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
def xp_path(sid): return state_path(sid).with_name("gamification.json")
def load_xp(sid):
    p=xp_path(sid)
    if not p.exists(): return 0
    data=json.loads(p.read_text(encoding="utf-8"))
    return int(data.get("xp",0))
def add_xp(sid, points):
    p=xp_path(sid)
    data={"xp":load_xp(sid),"last_activity":date.today().isoformat()}
    if p.exists():
        try:
            data=json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    total=max(0,int(data.get("xp",0))+int(points))
    data["xp"]=total
    data["last_activity"]=date.today().isoformat()
    xp_path(sid).write_text(
        json.dumps(data,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    return total
def ensure_state(sid):
    meta=load_meta(sid)
    if not meta.get("course_pdf"): raise HTTPException(400,"Ajoute d'abord le PDF du cours.")
    state=load_state(sid)
    if state is None:
        state=agent.initialiser_progression(str(subject_dir(sid)/meta["course_pdf"]))
        if "erreur" in state: raise HTTPException(400,state["erreur"])
        state["niveau_difficulte"]="Facile"
        state["exercices_corriges"]=0
        save_state(sid,state)
    elif "niveau_difficulte" not in state:
        state["niveau_difficulte"]="Facile"
        save_state(sid,state)
    return state

class SubjectIn(BaseModel): name:str
class ChatIn(BaseModel): question:str
class AnswerIn(BaseModel): answer:str
class StudyIn(BaseModel): level:str="Facile"
class AccountIn(BaseModel):
    name: str
    password: str
class FriendIn(BaseModel):
    name: str

def load_users():
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))

def save_users(users):
    USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()

def current_user(x_user_token: str | None = Header(default=None)):
    user_id = SESSIONS.get(x_user_token or "")
    if not user_id:
        return None
    return next((u for u in load_users() if u["id"] == user_id), None)

@app.get("/api/health")
def health(): return {"ok":True,"service":"Mentora"}
@app.get("/api/subjects")
def subjects():
    result=[]
    for p in SUBJECTS.iterdir():
        if p.is_dir() and (p/"subject.json").exists():
            m=json.loads((p/"subject.json").read_text(encoding="utf-8"))
            docs=m.get("documents",[])
            state=load_state(p.name)
            result.append({"id":p.name,"name":m["name"],"documents":docs,"progress":len(state.get("parties_validees",[])) if state else 0,"total":len(state.get("titres_chapitres",[])) if state else 0})
    return result
@app.post("/api/subjects")
def create_subject(data:SubjectIn):
    name=data.name.strip()
    if not name: raise HTTPException(400,"Nom requis")
    sid=uuid.uuid4().hex[:12]
    p=subject_dir(sid); p.mkdir(parents=True)
    for folder in ["Cours","TD","Diapositives","Autres"]: (p/folder).mkdir()
    save_meta(sid,{"name":name,"documents":[],"course_pdf":None,"created":date.today().isoformat()})
    return {"id":sid,"name":name}
@app.post("/api/subjects/{sid}/upload")
async def upload(sid:str, category:str="Autres", file:UploadFile=File(...)):
    meta=load_meta(sid)
    if category not in ["Cours","TD","Diapositives","Autres"]: category="Autres"
    filename=Path(file.filename or "document").name
    dest=subject_dir(sid)/category/filename
    # avoid overwriting an existing upload
    if dest.exists(): dest=dest.with_name(f"{dest.stem}_{uuid.uuid4().hex[:5]}{dest.suffix}")
    with dest.open("wb") as f: shutil.copyfileobj(file.file,f)
    item={"name":dest.name,"category":category,"path":str(dest.relative_to(subject_dir(sid))).replace("\\","/"),"size":dest.stat().st_size}
    meta["documents"].append(item)
    if category=="Cours" and dest.suffix.lower()==".pdf" and not meta.get("course_pdf"): meta["course_pdf"]=item["path"]
    save_meta(sid,meta)
    return item
@app.get("/api/subjects/{sid}/documents")
def documents(sid:str): return load_meta(sid).get("documents",[])
@app.delete("/api/subjects/{sid}")
def delete_subject(sid:str):
    directory=subject_dir(sid)
    if not directory.exists():
        raise HTTPException(404,"Matière introuvable")
    shutil.rmtree(directory)
    return {"deleted":True,"id":sid}
@app.delete("/api/subjects/{sid}/documents")
def delete_document(sid:str, path:str):
    meta=load_meta(sid)
    relative=Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400,"Chemin de document invalide")

    base=subject_dir(sid).resolve()
    file_path=(base/relative).resolve()
    if base not in file_path.parents:
        raise HTTPException(400,"Chemin de document invalide")

    document=next(
        (item for item in meta["documents"] if item["path"]==relative.as_posix()),
        None
    )
    if document is None:
        raise HTTPException(404,"Document introuvable")

    if file_path.exists():
        file_path.unlink()
    meta["documents"]=[item for item in meta["documents"] if item["path"]!=relative.as_posix()]
    if meta.get("course_pdf")==relative.as_posix():
        course=next(
            (
                item["path"] for item in meta["documents"]
                if item["category"]=="Cours"
                and item["path"].lower().endswith(".pdf")
            ),
            None
        )
        meta["course_pdf"]=course
    save_meta(sid,meta)
    return {"deleted":True,"path":relative.as_posix()}
@app.get("/api/subjects/{sid}/planning")
def planning(sid:str):
    state=ensure_state(sid)
    plan=agent.obtenir_planning_adaptatif(state); save_state(sid,state); return plan
@app.post("/api/subjects/{sid}/study")
def study(sid:str, data:StudyIn):
    state=ensure_state(sid); plan=agent.obtenir_planning_adaptatif(state)
    if plan.get("statut")=="termine": return {"planning":plan,"introduction":None}
    niveau=state.get("niveau_difficulte", data.level or "Facile")
    intro=None
    if plan.get("afficher_introduction"):
        chapter=plan["chapitre_actuel"]
        text=state.get("chapitres",{}).get(chapter,{}).get("texte","")
        intro=agent.generer_introduction_chapitre(text,plan.get("chapitre_precedent"),chapter)
    part=plan["partie_actuelle"]
    context=agent.contexte_de_la_partie(state,part)
    retriever=None
    if context is None:
        retriever=agent.obtenir_retriever(state["fichier_pdf"])
        context=agent._contexte_par_recherche(retriever,part)
    proposal=agent.proposer_exercices_apres_etude(state["fichier_pdf"],part,retriever=retriever,niveau=niveau,contexte=context)
    exs=proposal.get("exercices",[])
    question=(exs[0] if isinstance(exs[0],str) else exs[0].get("enonce",str(exs[0]))) if exs else None
    save_state(sid,state)
    return {"planning":plan,"introduction":intro,"exercise":{"question":question,"source":proposal.get("source","LLM"),"level":niveau,"attempt":1},"message":proposal.get("message","")}
@app.post("/api/subjects/{sid}/chat")
def chat(sid:str,data:ChatIn):
    state=ensure_state(sid)
    retriever=agent.obtenir_retriever(state["fichier_pdf"])
    try: answer=agent.poser_question_au_tuteur(retriever,data.question)
    except Exception as e: raise HTTPException(500,f"Le tuteur n'a pas pu répondre : {e}")
    return {"answer":answer}
@app.post("/api/subjects/{sid}/evaluate")
def evaluate(sid:str,data:AnswerIn):
    state=ensure_state(sid)
    # Exercise details are submitted from the active browser session.
    return {"note":"Évaluation attend une session d'exercice active."}
@app.post("/api/subjects/{sid}/evaluate-active")
def evaluate_active(sid:str, payload:dict):
    state=ensure_state(sid); q=payload.get("question",""); answer=payload.get("answer",""); attempt=int(payload.get("attempt",1))
    part=state["partie_actuelle"]; context=agent.contexte_de_la_partie(state,part)
    retriever=None
    if context is None:
        retriever=agent.obtenir_retriever(state["fichier_pdf"]); context=agent._contexte_par_recherche(retriever,part)
    result=agent.evaluer_etudiant(answer,q,part,attempt,retriever,context,4)
    if result.get("correct"):
        level=state.get("niveau_difficulte","Facile")
        total_xp=add_xp(sid,20)
        state["exercices_corriges"]=int(state.get("exercices_corriges",0))+1
        if level=="Facile":
            state["niveau_difficulte"]="Moyen"
            planning=agent.obtenir_planning_adaptatif(state)
            next_level="Moyen"
            completed=False
            xp=20
        elif level=="Moyen":
            state["niveau_difficulte"]="Difficile"
            planning=agent.obtenir_planning_adaptatif(state)
            next_level="Difficile"
            completed=False
            xp=20
        else:
            planning=agent.mettre_a_jour_progression(state,{"score_sur_20":20,"lacunes":"","maitrise_validee":True})
            total_xp=add_xp(sid,50)
            state["niveau_difficulte"]="Facile"
            next_level="Facile"
            completed=True
            xp=70
        save_state(sid,state)
        return {"correct":True,"score":20,"message":result.get("message","Bravo !"),"hint":"","solution":"","planning":planning,"xp":xp,"total_xp":total_xp,"completed":completed,"next_level":next_level}
    if attempt>=4:
        level=state.get("niveau_difficulte","Facile")
        planning=agent.obtenir_planning_adaptatif(state)
        next_question=agent.generer_exercice_remediation(
            retriever,
            part,
            result.get("solution",""),
            q,
            level,
            context
        )
        save_state(sid,state)
        return {"correct":False,"score":0,"message":"4 tentatives utilisées.","hint":"","solution":result.get("solution",""),"remediation_question":next_question,"attempt":1,"planning":planning,"xp":0,"total_xp":load_xp(sid),"completed":False,"next_level":level,"same_level":True}
    plan=agent.obtenir_planning_adaptatif(state); save_state(sid,state)
    remediation_question=agent.generer_exercice_remediation(
        retriever,
        part,
        result.get("hint",""),
        q,
        payload.get("level","Facile"),
        context
    )
    return {
        "correct":False,
        "score":0,
        "message":"",
        "hint":result.get("hint",""),
        "solution":"",
        "remediation_question":remediation_question,
        "attempt":attempt+1,
        "planning":plan,
        "xp":0,
        "total_xp":load_xp(sid),
        "completed":False
    }
@app.post("/api/subjects/{sid}/leave")
def leave_subject(sid:str):
    total=add_xp(sid,-10)
    return {"xp":-10,"total_xp":total}

@app.post("/api/auth/register")
def register(data:AccountIn):
    name=data.name.strip()
    if len(name)<2 or len(data.password)<4:
        raise HTTPException(400,"Nom trop court ou mot de passe trop court.")
    users=load_users()
    if any(u["name"].casefold()==name.casefold() for u in users):
        raise HTTPException(409,"Ce nom est déjà utilisé.")
    salt=secrets.token_hex(16)
    user={"id":uuid.uuid4().hex,"name":name,"salt":salt,"password":password_hash(data.password,salt),"friends":[],"xp":0,"projects":[]}
    users.append(user); save_users(users)
    token=secrets.token_urlsafe(24); SESSIONS[token]=user["id"]
    return {"token":token,"name":name,"xp":0,"friends":[],"level":1,"projects":[]}

@app.post("/api/auth/login")
def login(data:AccountIn):
    name=data.name.strip()
    user=next((u for u in load_users() if u["name"].casefold()==name.casefold()),None)
    if not user or password_hash(data.password,user["salt"])!=user["password"]:
        raise HTTPException(401,"Identifiants incorrects.")
    token=secrets.token_urlsafe(24); SESSIONS[token]=user["id"]
    return {"token":token,"name":user["name"],"xp":user.get("xp",0),"friends":user.get("friends",[]),"level":1+(int(user.get("xp",0))//100),"projects":user.get("projects",[])}

@app.post("/api/friends")
def add_friend(data:FriendIn, x_user_token: str | None = Header(default=None)):
    user=current_user(x_user_token)
    if not user:
        raise HTTPException(401,"Identifie-toi pour ajouter un ami.")
    name=data.name.strip()
    target=next((u for u in load_users() if u["name"].casefold()==name.casefold()),None)
    if not target:
        raise HTTPException(404,"Utilisateur introuvable.")
    if target["id"]==user["id"]:
        raise HTTPException(400,"Tu ne peux pas t'ajouter toi-même.")
    users=load_users()
    owner=next(u for u in users if u["id"]==user["id"])
    friend_ids=owner.setdefault("friends",[])
    if target["id"] not in friend_ids:
        friend_ids.append(target["id"])
    save_users(users)
    return {"friends":friend_ids}

@app.get("/api/gamification")
def gamification(sid:str|None=None, x_user_token: str | None = Header(default=None)):
    if sid:
        xp=load_xp(sid)
    else:
        xp=sum(
            load_xp(p.name)
            for p in SUBJECTS.iterdir()
            if p.is_dir()
        )
    level=1+(xp//100)
    user=current_user(x_user_token)
    streak=0
    monthly_bonus=False
    if user:
        streak=int(user.get("streak",0))
        month_key=date.today().strftime("%Y-%m")
        activity=user.setdefault("activity",{})
        today=date.today().isoformat()
        if today not in activity.get("days",[]):
            activity.setdefault("days",[]).append(today)
        month_days=[d for d in activity["days"] if d.startswith(month_key)]
        streak=len(set(month_days))
        if streak>=30:
            month_claimed=user.setdefault("monthly_claimed",[])
            if month_key not in month_claimed:
                xp+=30
                month_claimed.append(month_key)
                monthly_bonus=True
                streak=30
        user["xp"]=xp
        users=load_users()
        stored=next((u for u in users if u["id"]==user["id"]),None)
        if stored:
            stored["xp"]=xp
            stored["streak"]=streak
            stored["activity"]=activity
            stored["monthly_claimed"]=user.get("monthly_claimed",[])
            save_users(users)
    friends=[{"name":"Toi","xp":xp}]
    if user:
        users=load_users()
        owner=next(u for u in users if u["id"]==user["id"])
        for friend_id in owner.get("friends",[]):
            friend=next((u for u in users if u["id"]==friend_id),None)
            if friend:
                friends.append({"name":friend["name"],"xp":friend.get("xp",0)})
    projects=[]
    if level>=3:
        projects.append("Mini-projet : synthèse et application guidée du cours.")
    if level>=5:
        projects.append("Projet : construire une présentation pédagogique complète.")
    return {
        "xp":xp,
        "level":level,
        "streak":streak,
        "monthly_bonus":monthly_bonus,
        "friends":sorted(friends,key=lambda f:f["xp"],reverse=True),
        "projects":projects
    }
