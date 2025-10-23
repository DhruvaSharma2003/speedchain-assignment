from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile
from faster_whisper import WhisperModel
from fastapi import UploadFile, File
import os, uuid
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from memory import init_db, SessionLocal, get_or_create_session, add_message, upsert_entity, get_entities
import os, uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import json, os
from google import genai
from google.genai import types


MODEL = None
def get_whisper():
    global MODEL
    if MODEL is None:
        # tiny is fast & light; you can switch to "base" later
        MODEL = WhisperModel(
            os.getenv("WHISPER_MODEL", "tiny"),
            compute_type=os.getenv("WHISPER_COMPUTE", "int8")  # good CPU default
        )
    return MODEL


app = FastAPI(title="Speedchain Assignment API", version="0.0.1")

os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/calendar", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def _ics_escape(s: str) -> str:
    # Escape text per RFC 5545
    return s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")

def _fmt_ics_dt(dt: datetime) -> str:
    # UTC in YYYYMMDDTHHMMSSZ
    return dt.strftime("%Y%m%dT%H%M%SZ")

_gemini = None
def get_gemini():
    """Return a google-genai Client or None if no key."""
    global _gemini
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    if _gemini is None:
        _gemini = genai.Client(api_key=key)
    return _gemini

def _normalize_model_name(n: str) -> str:
    # API often returns "models/xxx" — both forms are accepted; keep short.
    return n.split("/", 1)[-1]

def choose_gemini_model(client) -> str:
    env = os.getenv("GEMINI_MODEL")
    if env:
        return env
    try:
        names = []
        for m in client.models.list():
            name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            if name:
                names.append(name)  # keep "models/..." EXACT
        # prefer flash variants
        for p in ["models/gemini-2.5-flash", "models/gemini-flash-latest"]:
            if p in names:
                return p
        # fallback: first gemini text model
        for n in names:
            if "gemini" in n and "embedding" not in n and "imagen" not in n and "veo" not in n:
                return n
    except Exception as e:
        print("Model discovery failed:", e)
    return "models/gemini-2.5-flash"


from google import genai
from google.genai import types

def _resp_to_text(resp) -> str | None:
    """
    Normalize google-genai response to plain text.
    Tries .text first, then walks candidates->content->parts.
    """
    t = getattr(resp, "text", None)
    if t:
        return t
    try:
        cands = getattr(resp, "candidates", []) or []
        for cand in cands:
            content = getattr(cand, "content", None) or (cand.get("content") if isinstance(cand, dict) else None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else None)
            if not parts:
                continue
            buf = []
            for p in parts:
                txt = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                if txt:
                    buf.append(txt)
            if buf:
                return "".join(buf)
    except Exception:
        pass
    return None

def plan_with_gemini(user_text: str, entities: dict):
    """
    Ask Gemini to extract intent/slots and draft a reply (JSON).
    Returns dict or None to fall back to rule-based.
    """
    client = get_gemini()
    if client is None:
        return None

    system = (
        "You are a concise scheduling assistant. "
        "Always respond with a single compact JSON object ONLY, no extra words. "
        "Schema:{\"intent\":\"smalltalk|inquire_service|schedule_appointment|reschedule|cancel\","
        "\"slots\":{\"name\":\"\",\"email\":\"\",\"date\":\"YYYY-MM-DD\",\"time\":\"HH:MM\"},"
        "\"reply\":\"short helpful reply\"}. "
        "If a slot is unknown, use empty string."
    )

    # Candidate list: prefer explicit env, else auto-discover, else a safe default
    candidates = []
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        candidates.append(env_model)
    else:
        try:
            names = []
            for m in client.models.list():
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
                if name:
                    names.append(name)  # keep "models/..." exact
            # prefer flash (chat) models
            for p in ["models/gemini-2.5-flash", "models/gemini-flash-latest"]:
                if p in names:
                    candidates.append(p)
                    break
            if not candidates and names:
                # pick any gemini text model as last resort
                for n in names:
                    if "gemini" in n and "embedding" not in n and "imagen" not in n and "veo" not in n:
                        candidates.append(n)
                        break
        except Exception as e:
            print("Model discovery failed:", e)
    if not candidates:
        candidates = ["models/gemini-2.5-flash"]

    for model_name in candidates:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=(
                    f"Known entities: {json.dumps(entities)}\n"
                    f"User said: {user_text}\n"
                    f"Return ONLY the JSON object."
                ),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            raw = _resp_to_text(resp)
            if not raw:
                raise ValueError("empty model response")
            data = json.loads(raw)
            if isinstance(data, dict) and "reply" in data:
                data.setdefault("slots", {})
                return data
        except Exception as e:
            print(f"Gemini error with {model_name}: {e}")

    return None  # fall back to rule-based


init_db()

# serve generated audio files
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/debug/models")
def list_gemini_models():
    c = get_gemini()
    if not c:
        return {"gemini_available": False, "models": []}
    names = []
    for m in c.models.list():
        name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
        if name:
            names.append(name)  # e.g. "models/gemini-1.5-flash-8b"
    return {"gemini_available": True, "models": names}


@app.get("/health")
def health():
    return {"ok": True, "service": "api", "version": "0.0.1"}

from fastapi import UploadFile, File

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Transcribe short audio using faster-whisper with AUTO language detection.
    You can force a language by setting env WHISPER_LANG (e.g., "hi").
    """
    import os, tempfile
    audio_bytes = await file.read()

    # write to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model = get_whisper()
    # Auto-detect unless WHISPER_LANG is set (e.g., hi, en, fr)
    force_lang = os.getenv("WHISPER_LANG") or None

    segments, info = model.transcribe(
        tmp_path,
        beam_size=1,
        language=force_lang,        # None => auto-detect
        task="transcribe"           # don't translate; keep original language
    )

    text = "".join(seg.text for seg in segments).strip()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return {
        "text": text,
        "language_detected": getattr(info, "language", None),
        "duration_sec": getattr(info, "duration", None),
        "notes": f"Model={os.getenv('WHISPER_MODEL','tiny')} compute={os.getenv('WHISPER_COMPUTE','int8')} forced_lang={force_lang}"
    }

class TTSIn(BaseModel):
    text: str
    lang: str | None = "en"   # keep English by default

@app.post("/tts")
async def tts(body: TTSIn):
    """
    Simple TTS using gTTS. Saves an MP3 under /static/audio and returns its path.
    Frontend will prefix the backend base URL.
    """
    text = (body.text or "").strip()
    if not text:
        return {"error": "text is required"}

    lang = body.lang or "en"
    fname = f"audio/tts_{uuid.uuid4().hex}.mp3"
    full_path = os.path.join("static", fname)

    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(full_path)

    return {"audio_path": f"/static/{fname}"}

class ScheduleIn(BaseModel):
    session_id: str
    title: str | None = "Appointment"
    date: str               # "YYYY-MM-DD"
    time: str               # "HH:MM" 24h
    duration_min: int | None = 30
    timezone: str | None = "Asia/Kolkata"
    email: str | None = None   # optional; email sending only if SMTP env vars are set

@app.post("/schedule")
def schedule(body: ScheduleIn):
    """
    Create an .ics calendar invite and a dummy Meet link.
    If SMTP env vars are present and 'email' is given, we try to email it (best-effort).
    Returns JSON with the ICS path and link either way.
    """
    # Parse local datetime
    try:
        y, m, d = map(int, body.date.split("-"))
        hh, mm = map(int, body.time.split(":"))
    except Exception:
        return {"error": "Invalid date/time format. Expect date=YYYY-MM-DD and time=HH:MM"}

    tz = ZoneInfo(body.timezone or "UTC")
    start_local = datetime(y, m, d, hh, mm, tzinfo=tz)
    end_local = start_local + timedelta(minutes=body.duration_min or 30)
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))

    # Dummy video link
    meet_link = f"https://meet.jit.si/speedchain-{uuid.uuid4().hex[:8]}"

    # Build ICS text (simple & standards-compliant)
    uid = f"{uuid.uuid4()}@speedchain"
    dtstamp = _fmt_ics_dt(datetime.utcnow())
    dtstart = _fmt_ics_dt(start_utc)
    dtend = _fmt_ics_dt(end_utc)
    title = body.title or "Appointment"
    desc = f"Booked via Speedchain demo. Join: {meet_link}"
    ics_text = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//speedchain//assignment//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{dtstamp}\r\n"
        f"DTSTART:{dtstart}\r\n"
        f"DTEND:{dtend}\r\n"
        f"SUMMARY:{_ics_escape(title)}\r\n"
        f"DESCRIPTION:{_ics_escape(desc)}\r\n"
        f"LOCATION:{_ics_escape(meet_link)}\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )

    # Save ICS
    fname = f"calendar/{uuid.uuid4().hex}.ics"
    full_path = os.path.join("static", fname)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(ics_text)

    # Optional: email it if SMTP env present and email was provided
    sent = False
    err = None
    if body.email and all(os.getenv(k) for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "FROM_EMAIL"]):
        try:
            import smtplib
            from email.message import EmailMessage
            with open(full_path, "rb") as f:
                ics_bytes = f.read()
            msg = EmailMessage()
            msg["Subject"] = f"{title} — confirmation"
            msg["From"] = os.getenv("FROM_EMAIL")
            msg["To"] = body.email
            msg.set_content(f"Your appointment is confirmed.\nJoin: {meet_link}\n\n(ICS attached)")
            msg.add_attachment(ics_bytes, maintype="text", subtype="calendar", filename="invite.ics", params={"method":"PUBLISH"})
            with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT"))) as s:
                s.starttls()
                s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
                s.send_message(msg)
            sent = True
        except Exception as e:
            err = f"Email failed: {e!s}"

    return {
        "ok": True,
        "meet_link": meet_link,
        "ics_path": f"/static/{fname}",
        "start_local": start_local.isoformat(),
        "end_local": end_local.isoformat(),
        "timezone": body.timezone or "UTC",
        "email_sent": sent,
        "email_error": err
    }



class ChatIn(BaseModel):
    session_id: str
    user_text: str

class ChatOut(BaseModel):
    text: str
    entities: Dict[str, str] = {}

def simple_nlu(user_text: str) -> Dict[str, str]:
    """
    Tiny heuristic extractor:
    - Captures 'my name is <X>' or 'I am <X>' → entity name
    - Captures 'email is <x@x>' → entity email
    - Detects scheduling phrases (we'll wire scheduling next)
    """
    import re
    slots: Dict[str, str] = {}
    m = re.search(r"\bmy name is ([A-Za-z][A-Za-z .'-]{1,40})", user_text, re.I)
    if not m:
        m = re.search(r"\bi am ([A-Za-z][A-Za-z .'-]{1,40})", user_text, re.I)
    if m:
        slots["name"] = m.group(1).strip()

    m = re.search(r"\b(?:email|mail)\s*(?:is|:)?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", user_text, re.I)
    if m:
        slots["email"] = m.group(1).strip()

    # very rough schedule detection (just to showcase memory)
    if re.search(r"\b(schedule|book|appointment|meeting)\b", user_text, re.I):
        slots["intent"] = "schedule_appointment"
    return slots

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    db = SessionLocal()
    try:
        sess = get_or_create_session(db, body.session_id)
        add_message(db, sess.id, role="user", text=body.user_text)

        # current memory to aid the model
        ents = get_entities(db, sess.id)

        plan = plan_with_gemini(body.user_text, ents)  # None if no key or error

        if plan is None:
            # ---- fallback: simple rule-based NLU ----
            slots = simple_nlu(body.user_text)
            for k, v in slots.items():
                if k != "intent" and v:
                    upsert_entity(db, sess.id, k, v)
            ents = get_entities(db, sess.id)
            name = ents.get("name")
            if "intent" in slots and slots["intent"] == "schedule_appointment":
                reply = "Got it — you'd like to schedule an appointment. Share your preferred date, time, and email."
            else:
                reply = f"Hi {name}! How can I help you today?" if name \
                        else "Hi! How can I help you today? (You can tell me your name with 'my name is …')"
        else:
            # ---- Gemini path: upsert extracted slots and use its reply ----
            for k, v in (plan.get("slots") or {}).items():
                if k and v:
                    upsert_entity(db, sess.id, k, v)
            ents = get_entities(db, sess.id)
            reply = plan.get("reply", "Okay.")

        add_message(db, sess.id, role="assistant", text=reply)
        return ChatOut(text=reply, entities=ents)
    finally:
        db.close()





