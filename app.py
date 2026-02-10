import os
import re
import io
import json
import time
import uuid
import wave
import shutil
import threading
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import gradio as gr

# Gemini SDK
from google import genai
from google.genai import types

# Optional: Whisper (heavy). You can switch to faster-whisper later.
import whisper


# =========================
# Config
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    # We'll still start UI, but job will fail with a clear error.
    pass

# Defaults (change in UI too)
DEFAULT_VIDEO_MODEL = "gemini-3-flash-preview"  # Video understanding examples show gemini-3-flash-preview :contentReference[oaicite:2]{index=2}
DEFAULT_TEXT_MODEL = "gemini-3-flash-preview"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS docs example :contentReference[oaicite:3]{index=3}
DEFAULT_TTS_VOICE = "Kore"  # TTS docs example voice name :contentReference[oaicite:4]{index=4}

BASE_DIR = "/tmp/vdo_jobs"
os.makedirs(BASE_DIR, exist_ok=True)

# Concurrency guard (web-only plan safe)
RUN_LOCK = threading.Semaphore(1)


# =========================
# Helpers
# =========================
def now_ts() -> float:
    return time.time()

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "file"

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")

def write_wave_file(path: str, pcm_bytes: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)

def zip_dir(folder: str, out_zip: str) -> None:
    base = out_zip[:-4] if out_zip.lower().endswith(".zip") else out_zip
    shutil.make_archive(base, "zip", folder)


# =========================
# Job state
# =========================
@dataclass
class Job:
    id: str
    created_at: float
    status: str  # queued/running/done/error
    progress: int  # 0-100
    message: str
    out_dir: str
    out_zip: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

JOBS: Dict[str, Job] = {}


def set_job(job_id: str, **kwargs):
    job = JOBS.get(job_id)
    if not job:
        return
    for k, v in kwargs.items():
        setattr(job, k, v)

def job_to_dict(job: Job) -> Dict[str, Any]:
    d = asdict(job)
    # Don’t leak big internal paths
    return d


# =========================
# Gemini clients
# =========================
def gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_generate_text(model: str, prompt: str) -> str:
    client = gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (resp.text or "").strip()

def gemini_youtube_video_to_script(
    model: str,
    youtube_url: str,
    prompt: str,
    start_s: Optional[int] = None,
    end_s: Optional[int] = None,
    fps: Optional[float] = None,
) -> str:
    """
    Uses YouTube URL as file_data in prompt. :contentReference[oaicite:5]{index=5}
    """
    client = gemini_client()

    video_md = None
    if start_s is not None or end_s is not None or fps is not None:
        # VideoMetadata supports clipping intervals and fps sampling (docs show start_offset/end_offset/fps). :contentReference[oaicite:6]{index=6}
        kwargs = {}
        if start_s is not None:
            kwargs["start_offset"] = f"{start_s}s"
        if end_s is not None:
            kwargs["end_offset"] = f"{end_s}s"
        if fps is not None:
            kwargs["fps"] = fps
        video_md = types.VideoMetadata(**kwargs)

    parts = [
        types.Part(
            file_data=types.FileData(file_uri=youtube_url),
            video_metadata=video_md if video_md else None,
        ),
        types.Part(text=prompt),
    ]

    resp = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
    )
    return (resp.text or "").strip()

def gemini_tts_wav(
    tts_model: str,
    text: str,
    voice_name: str = DEFAULT_TTS_VOICE,
) -> bytes:
    """
    Native TTS via Gemini API. Response modality AUDIO + SpeechConfig + voice_name. :contentReference[oaicite:7]{index=7}
    Returns PCM bytes (24kHz) that can be saved to WAV.
    """
    client = gemini_client()
    resp = client.models.generate_content(
        model=tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        ),
    )
    data = resp.candidates[0].content.parts[0].inline_data.data
    return data


# =========================
# Prompts
# =========================
def build_script_prompt(
    analysis_mode: str,
    out_lang: str,
    style: str,
    length_hint: str,
    include_srt: bool,
    include_hashtags: bool,
    include_titles: bool,
) -> str:
    """
    Returns a single prompt that asks model to output a clean result.
    We keep it simple to avoid fragile JSON parsing.
    """
    srt_req = (
        "\n- Also output SRT captions (timestamps)."
        "\n  Format:\n"
        "  [SRT]\n"
        "  1\n"
        "  00:00:00,000 --> 00:00:03,000\n"
        "  ...\n"
        "  [/SRT]\n"
    ) if include_srt else ""

    hashtags_req = "\n- Also output 20 relevant hashtags. Put under [HASHTAGS]...[/HASHTAGS]" if include_hashtags else ""
    titles_req = "\n- Also output 10 title ideas + 10 thumbnail text ideas under [TITLES]...[/TITLES]" if include_titles else ""

    mode_rules = ""
    if analysis_mode == "Motion/Activity (scene-by-scene)":
        mode_rules = (
            "- Focus on visual actions and activities. Break down scene-by-scene.\n"
            "- Write narration that describes what happens, step by step.\n"
            "- If there is little/no speech, still produce a complete narration.\n"
        )
    else:
        mode_rules = (
            "- Focus on spoken content and key story points.\n"
            "- Produce a clean narration script for voice-over.\n"
        )

    prompt = f"""
You are a professional video analyst + scriptwriter + translator.
Write in {out_lang}. Style: {style}. Target length: {length_hint}.

OUTPUT FORMAT (IMPORTANT):
- Put the final narration script between:
  [SCRIPT]
  ...
  [/SCRIPT]

- If you provide scene bullets, put them between:
  [SCENES]
  - 00:00-00:10 ...
  [/SCENES]

Rules:
{mode_rules}
- Do NOT add extra labels outside the required blocks.
{srt_req}
{hashtags_req}
{titles_req}
""".strip()
    return prompt

def build_from_transcript_prompt(script_prompt: str, transcript_text: str) -> str:
    return f"""{script_prompt}

Here is the transcript / notes:
[TRANSCRIPT]
{transcript_text}
[/TRANSCRIPT]
"""

def extract_block(text: str, tag: str) -> str:
    """
    Extract content between [TAG] and [/TAG].
    """
    pattern = re.compile(rf"\[{re.escape(tag)}\](.*?)\[/\s*{re.escape(tag)}\s*\]", re.S | re.I)
    m = pattern.search(text or "")
    return (m.group(1).strip() if m else "")

def simple_whisper_transcribe(audio_path: str, whisper_model: str = "small") -> Dict[str, Any]:
    model = whisper.load_model(whisper_model)
    result = model.transcribe(audio_path)
    return result

def whisper_segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    def fmt_ts(t: float) -> str:
        ms = int(round(t * 1000))
        hh = ms // 3600000; ms %= 3600000
        mm = ms // 60000; ms %= 60000
        ss = ms // 1000; ms %= 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        start = fmt_ts(float(seg.get("start", 0)))
        end = fmt_ts(float(seg.get("end", 0)))
        txt = (seg.get("text", "") or "").strip()
        if not txt:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(txt)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# =========================
# Pipeline
# =========================
def pipeline_run(job_id: str, payload: Dict[str, Any]) -> None:
    """
    Runs inside a background thread.
    """
    with RUN_LOCK:
        job = JOBS[job_id]
        set_job(job_id, status="running", progress=1, message="Starting...")

        out_dir = job.out_dir
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Payload
            use_youtube_direct = bool(payload.get("use_youtube_direct"))
            url = (payload.get("url") or "").strip()
            uploaded_path = payload.get("uploaded_path")  # already saved file
            analysis_mode = payload.get("analysis_mode")
            out_lang = payload.get("out_lang")
            style = payload.get("style")
            length_hint = payload.get("length_hint")

            include_srt = bool(payload.get("include_srt"))
            include_hashtags = bool(payload.get("include_hashtags"))
            include_titles = bool(payload.get("include_titles"))

            video_model = (payload.get("video_model") or DEFAULT_VIDEO_MODEL).strip()
            text_model = (payload.get("text_model") or DEFAULT_TEXT_MODEL).strip()

            # Video metadata options for direct YouTube analysis
            start_s = payload.get("start_s")
            end_s = payload.get("end_s")
            fps = payload.get("fps")

            gen_tts = bool(payload.get("gen_tts"))
            tts_model = (payload.get("tts_model") or DEFAULT_TTS_MODEL).strip()
            tts_voice = (payload.get("tts_voice") or DEFAULT_TTS_VOICE).strip()

            # Build prompt
            base_prompt = build_script_prompt(
                analysis_mode=analysis_mode,
                out_lang=out_lang,
                style=style,
                length_hint=length_hint,
                include_srt=include_srt,
                include_hashtags=include_hashtags,
                include_titles=include_titles,
            )

            # -----------------------
            # PATH A: YouTube direct
            # -----------------------
            if use_youtube_direct:
                if not url or "youtube.com" not in url and "youtu.be" not in url:
                    raise ValueError("YouTube direct option requires a YouTube URL.")

                set_job(job_id, progress=10, message="Gemini analyzing YouTube video directly...")
                prompt = base_prompt + "\n\nTask: Analyze the provided video and produce the requested outputs."
                raw = gemini_youtube_video_to_script(
                    model=video_model,
                    youtube_url=url,
                    prompt=prompt,
                    start_s=int(start_s) if start_s not in (None, "") else None,
                    end_s=int(end_s) if end_s not in (None, "") else None,
                    fps=float(fps) if fps not in (None, "") else None,
                )
                set_job(job_id, progress=60, message="Extracting outputs...")
                script_text = extract_block(raw, "SCRIPT") or raw
                scenes_text = extract_block(raw, "SCENES")
                srt_text = extract_block(raw, "SRT")
                hashtags_text = extract_block(raw, "HASHTAGS")
                titles_text = extract_block(raw, "TITLES")

            # -----------------------
            # PATH B: Local download + whisper
            # -----------------------
            else:
                # Acquire video file
                set_job(job_id, progress=5, message="Preparing video input...")
                input_video = None

                if uploaded_path:
                    input_video = uploaded_path
                elif url:
                    set_job(job_id, progress=8, message="Downloading video (yt-dlp)...")
                    input_video = os.path.join(out_dir, "input.mp4")
                    # yt-dlp output template
                    # Use best mp4 where possible
                    cmd = [
                        "yt-dlp",
                        "-f", "bv*+ba/b",
                        "--merge-output-format", "mp4",
                        "-o", input_video,
                        url
                    ]
                    run_cmd(cmd)
                else:
                    raise ValueError("Provide a URL or upload a video.")

                set_job(job_id, progress=18, message="Extracting audio (ffmpeg)...")
                audio_path = os.path.join(out_dir, "audio.wav")
                run_cmd(["ffmpeg", "-y", "-i", input_video, "-vn", "-ac", "1", "-ar", "16000", audio_path])

                set_job(job_id, progress=30, message="Transcribing (Whisper)...")
                whisper_model = payload.get("whisper_model") or "small"
                tr = simple_whisper_transcribe(audio_path, whisper_model=whisper_model)

                transcript_text = (tr.get("text") or "").strip()
                segments = tr.get("segments") or []

                # optional srt from whisper segments
                whisper_srt = whisper_segments_to_srt(segments) if include_srt else ""

                set_job(job_id, progress=50, message="Generating script via Gemini (from transcript)...")
                full_prompt = build_from_transcript_prompt(base_prompt, transcript_text)
                raw = gemini_generate_text(text_model, full_prompt)

                set_job(job_id, progress=70, message="Extracting outputs...")
                script_text = extract_block(raw, "SCRIPT") or raw
                scenes_text = extract_block(raw, "SCENES")
                srt_text = extract_block(raw, "SRT") or whisper_srt
                hashtags_text = extract_block(raw, "HASHTAGS")
                titles_text = extract_block(raw, "TITLES")

                # Save transcript too
                with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                    f.write(transcript_text + "\n")

            # Save script outputs
            with open(os.path.join(out_dir, "script.txt"), "w", encoding="utf-8") as f:
                f.write(script_text.strip() + "\n")

            if scenes_text.strip():
                with open(os.path.join(out_dir, "scenes.txt"), "w", encoding="utf-8") as f:
                    f.write(scenes_text.strip() + "\n")

            if srt_text.strip():
                with open(os.path.join(out_dir, "captions.srt"), "w", encoding="utf-8") as f:
                    f.write(srt_text.strip() + "\n")

            if hashtags_text.strip():
                with open(os.path.join(out_dir, "hashtags.txt"), "w", encoding="utf-8") as f:
                    f.write(hashtags_text.strip() + "\n")

            if titles_text.strip():
                with open(os.path.join(out_dir, "titles_thumbnails.txt"), "w", encoding="utf-8") as f:
                    f.write(titles_text.strip() + "\n")

            # TTS (optional)
            audio_out = None
            if gen_tts:
                set_job(job_id, progress=80, message="Generating speech audio (Gemini TTS)...")
                # TTS best prompt: you can steer style by prefixing
                tts_input = script_text.strip()
                pcm = gemini_tts_wav(tts_model, tts_input, voice_name=tts_voice)
                audio_out = os.path.join(out_dir, "voice.wav")
                write_wave_file(audio_out, pcm, channels=1, rate=24000, sample_width=2)

            # Metadata
            meta = {
                "job_id": job_id,
                "created_at": job.created_at,
                "use_youtube_direct": use_youtube_direct,
                "url": url,
                "analysis_mode": analysis_mode,
                "out_lang": out_lang,
                "style": style,
                "length_hint": length_hint,
                "models": {
                    "video_model": video_model,
                    "text_model": text_model,
                    "tts_model": tts_model if gen_tts else None,
                    "tts_voice": tts_voice if gen_tts else None,
                },
                "outputs": {
                    "script": "script.txt",
                    "scenes": "scenes.txt" if scenes_text.strip() else None,
                    "captions": "captions.srt" if srt_text.strip() else None,
                    "hashtags": "hashtags.txt" if hashtags_text.strip() else None,
                    "titles": "titles_thumbnails.txt" if titles_text.strip() else None,
                    "audio": "voice.wav" if audio_out else None,
                },
            }
            with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Zip outputs
            set_job(job_id, progress=95, message="Packaging outputs...")
            out_zip = os.path.join(out_dir, "outputs.zip")
            zip_dir(out_dir, out_zip)

            set_job(job_id, status="done", progress=100, message="Done.", out_zip=out_zip, result=meta)

        except Exception as e:
            set_job(job_id, status="error", progress=100, message="Error", error=str(e))


def create_job(payload: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(BASE_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    job = Job(
        id=job_id,
        created_at=now_ts(),
        status="queued",
        progress=0,
        message="Queued",
        out_dir=out_dir,
    )
    JOBS[job_id] = job

    # Run in background thread
    th = threading.Thread(target=pipeline_run, args=(job_id, payload), daemon=True)
    th.start()
    return job_id


# =========================
# FastAPI
# =========================
app = FastAPI(title="VDO Script + Gemini TTS (Web-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": now_ts()}

@app.post("/api/jobs")
def api_create_job(payload: Dict[str, Any]):
    job_id = create_job(payload)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job_to_dict(job))

@app.get("/api/jobs/{job_id}/download")
def api_download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.out_zip or not os.path.exists(job.out_zip):
        raise HTTPException(status_code=400, detail="Job not ready")
    return FileResponse(job.out_zip, filename=f"{job_id}_outputs.zip")


# =========================
# Gradio UI
# =========================
def ui_submit(
    url: str,
    video_file,
    use_youtube_direct: bool,
    analysis_mode: str,
    out_lang: str,
    style: str,
    length_hint: str,
    include_srt: bool,
    include_hashtags: bool,
    include_titles: bool,
    video_model: str,
    text_model: str,
    start_s,
    end_s,
    fps,
    use_tts: bool,
    tts_model: str,
    tts_voice: str,
    whisper_model: str,
):
    uploaded_path = None
    if video_file is not None:
        # Save uploaded video into job folder later; for now save to a temp file.
        tmp_dir = os.path.join(BASE_DIR, "_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        name = safe_filename(os.path.basename(video_file.name))
        uploaded_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex[:8]}_{name}")
        shutil.copy(video_file.name, uploaded_path)

    payload = {
        "url": (url or "").strip(),
        "uploaded_path": uploaded_path,
        "use_youtube_direct": bool(use_youtube_direct),
        "analysis_mode": analysis_mode,
        "out_lang": out_lang,
        "style": style,
        "length_hint": length_hint,
        "include_srt": bool(include_srt),
        "include_hashtags": bool(include_hashtags),
        "include_titles": bool(include_titles),
        "video_model": video_model.strip() if video_model else DEFAULT_VIDEO_MODEL,
        "text_model": text_model.strip() if text_model else DEFAULT_TEXT_MODEL,
        "start_s": start_s,
        "end_s": end_s,
        "fps": fps,
        "gen_tts": bool(use_tts),
        "tts_model": tts_model.strip() if tts_model else DEFAULT_TTS_MODEL,
        "tts_voice": tts_voice.strip() if tts_voice else DEFAULT_TTS_VOICE,
        "whisper_model": whisper_model.strip() if whisper_model else "small",
    }
    job_id = create_job(payload)
    return job_id, f"Job created: {job_id}"

def ui_poll(job_id: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return "No job id", 0, "", None

    job = JOBS.get(job_id)
    if not job:
        return "Job not found", 0, "", None

    status_line = f"Status: {job.status} | {job.progress}% | {job.message}"
    err = job.error or ""
    download_link = None
    if job.status == "done":
        download_link = f"/api/jobs/{job_id}/download"
    return status_line, job.progress, err, download_link


with gr.Blocks(title="VDO → Script → (Optional) Gemini TTS") as demo:
    gr.Markdown(
        """
# VDO / Social Link → Script (Optional: YouTube direct) → (Optional) Gemini Generated Speech
**Web-only (Render Web Service)**. UptimeRobot: ping `/health`.
- YouTube direct video-understanding uses `file_data=file_uri=<youtube-url>` as in official docs. :contentReference[oaicite:8]{index=8}
- Native TTS uses response modality AUDIO + SpeechConfig with voice_name. :contentReference[oaicite:9]{index=9}
        """.strip()
    )

    with gr.Tab("1) Input"):
        url = gr.Textbox(label="Video / Social URL (YouTube/TikTok/FB/etc.)", placeholder="https://...")
        video_file = gr.File(label="Or Upload Video (mp4/mov)", file_types=[".mp4", ".mov", ".mkv", ".webm"])
        use_youtube_direct = gr.Checkbox(
            label="Option: Use YouTube URL directly with Gemini (no download/whisper)",
            value=False
        )
        gr.Markdown("If enabled, URL must be a **public YouTube** link. (Preview feature per docs.) :contentReference[oaicite:10]{index=10}")

        with gr.Row():
            start_s = gr.Number(label="(Optional) Clip start (seconds)", value=None, precision=0)
            end_s = gr.Number(label="(Optional) Clip end (seconds)", value=None, precision=0)
            fps = gr.Number(label="(Optional) FPS sampling (e.g. 0.5, 1, 2)", value=None)

    with gr.Tab("2) Script Options"):
        analysis_mode = gr.Dropdown(
            ["Direct Script (transcript-driven)", "Motion/Activity (scene-by-scene)"],
            value="Direct Script (transcript-driven)",
            label="Analysis Mode",
        )
        out_lang = gr.Textbox(label="Output Language", value="Myanmar (Burmese)")
        style = gr.Dropdown(
            ["Movie recap", "Documentary calm", "UGC ad", "Storytelling dramatic", "Neutral"],
            value="Movie recap",
            label="Style preset",
        )
        length_hint = gr.Dropdown(
            ["Short (30–60s)", "1–3 minutes", "5–10 minutes", "Custom (model decides)"],
            value="1–3 minutes",
            label="Length",
        )

        with gr.Row():
            include_srt = gr.Checkbox(label="Include SRT captions (best-effort)", value=True)
            include_hashtags = gr.Checkbox(label="Include hashtags", value=True)
            include_titles = gr.Checkbox(label="Include title + thumbnail ideas", value=True)

    with gr.Tab("3) Models"):
        video_model = gr.Textbox(label="Video model (YouTube direct)", value=DEFAULT_VIDEO_MODEL)
        text_model = gr.Textbox(label="Text model (script from transcript)", value=DEFAULT_TEXT_MODEL)
        whisper_model = gr.Dropdown(["tiny", "base", "small", "medium"], value="small", label="Whisper model (local path)")

    with gr.Tab("4) Optional: Generated Speech (Gemini TTS)"):
        use_tts = gr.Checkbox(label="Generate speech audio (WAV)", value=False)
        tts_model = gr.Textbox(label="TTS model", value=DEFAULT_TTS_MODEL)
        tts_voice = gr.Textbox(label="Voice name (optional)", value=DEFAULT_TTS_VOICE)
        gr.Markdown("TTS uses Gemini API response_modality=AUDIO and SpeechConfig (Preview). :contentReference[oaicite:11]{index=11}")

    with gr.Row():
        btn = gr.Button("Run Job")
        job_id_box = gr.Textbox(label="Job ID", interactive=False)
        msg = gr.Textbox(label="Message", interactive=False)

    with gr.Tab("5) Progress / Download"):
        status = gr.Textbox(label="Status", interactive=False)
        prog = gr.Slider(label="Progress", minimum=0, maximum=100, step=1, interactive=False)
        err = gr.Textbox(label="Error (if any)", interactive=False)
        download = gr.Textbox(label="Download URL (open in browser)", interactive=False)

        poll_btn = gr.Button("Refresh status")

    btn.click(
        ui_submit,
        inputs=[
            url, video_file, use_youtube_direct,
            analysis_mode, out_lang, style, length_hint,
            include_srt, include_hashtags, include_titles,
            video_model, text_model,
            start_s, end_s, fps,
            use_tts, tts_model, tts_voice,
            whisper_model
        ],
        outputs=[job_id_box, msg]
    )

    poll_btn.click(ui_poll, inputs=[job_id_box], outputs=[status, prog, err, download])

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
