# ui/app.py
import os
import requests
import gradio as gr

def _join(base, path):
    base = base.rstrip("/")
    return base + path

def ping_backend(backend_url):
    try:
        r = requests.get(_join(backend_url, "/health"), timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def transcribe(audio_path, backend_url):
    if not audio_path:
        return gr.update(value="", placeholder="No audio recorded")
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
            r = requests.post(_join(backend_url, "/stt"), files=files, timeout=60)
        data = r.json()
        return data.get("text", "") or ""
    except Exception as e:
        return f"[STT error] {e}"

def send_text(message, session_id, backend_url, history):
    if not message.strip():
        return history, "", None
    try:
        # 1) send to /chat
        r = requests.post(_join(backend_url, "/chat"), json={
            "session_id": session_id,
            "user_text": message.strip()
        }, timeout=60)
        data = r.json()
        bot = data.get("text", "Okay.")

        # 2) append to history
        history = history + [(message, bot)]

        # 3) TTS for the bot reply
        tts_r = requests.post(_join(backend_url, "/tts"), json={"text": bot, "lang": "en"}, timeout=60)
        tts = tts_r.json()
        audio_path = tts.get("audio_path")
        audio_url = _join(backend_url, audio_path) if audio_path else None

        return history, "", audio_url
    except Exception as e:
        history = history + [(message, f"[Error] {e}")]
        return history, "", None

with gr.Blocks(title="Gloss & Glow — AI Receptionist") as demo:
    gr.Markdown("## Gloss & Glow — AI Receptionist\nVoice in → Bookings out.\n")

    with gr.Row():
        backend_url = gr.Textbox(label="Backend URL (port 8000)", value="", placeholder="https://xxxxx-8000.app.github.dev")
        session_id = gr.Textbox(label="Session ID", value="demo-1", scale=0)
        ping_btn = gr.Button("Ping /health", variant="secondary")
    ping_out = gr.JSON(label="Backend status")

    chatbot = gr.Chatbot(label="Chat", height=350)

    with gr.Row():
        msg_tb = gr.Textbox(label="Message", placeholder="Say hi or ask to book…", scale=3)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Markdown("### Mic → STT")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Record a short message")
        stt_btn = gr.Button("Transcribe & Fill")
    # when we have the reply, we’ll speak it:
    tts_player = gr.Audio(label="AI Voice Reply (auto after bot answers)", autoplay=True)

    # wiring
    ping_btn.click(ping_backend, inputs=[backend_url], outputs=[ping_out])
    stt_btn.click(transcribe, inputs=[mic, backend_url], outputs=[msg_tb])
    send_btn.click(send_text, inputs=[msg_tb, session_id, backend_url, chatbot], outputs=[chatbot, msg_tb, tts_player])

if __name__ == "__main__":
    # Gradio auto picks a port; set 7860 for Codespaces clarity
    demo.launch(server_name="0.0.0.0", server_port=7860)
