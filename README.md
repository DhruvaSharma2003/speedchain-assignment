# speedchain-assignment

Voice AI receptionist: **STT → LLM (Gemini) → TTS**, plus **memory** and **scheduling (.ics + meet link)**.  
Frontend (Gradio) + Backend (FastAPI). Runs entirely in GitHub Codespaces.

---

## ✨ Features
- 🎤 **STT**: browser mic → `/stt` (Whisper via `faster-whisper`) or stub; English-first.
- 🧠 **LLM**: `/chat` uses **Gemini** when `GOOGLE_API_KEY` is set; **automatic fallback** to rule-based when not.
- 📝 **Memory**: SQLite remembers `name`, `email`, etc. per `session_id`.
- 🔊 **TTS**: `/tts` uses gTTS; audio served from `/static/audio/...`.
- 📅 **Scheduling**: `/schedule` generates `.ics` and a dummy meet link; optional email send if SMTP configured.
- 🖥️ **Frontend**: simple page to Ping, record mic, chat, auto-speak, and create calendar invites.

---

