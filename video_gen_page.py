"""
Video Generation Page — OpenRouter Video Models
================================================
Full-featured video generation page mirroring app.py functionality:
- Prompt input with LLM enhancer (video-specific system prompts)
- Prompt manager (load from YAML/JSON)
- All OpenRouter video models with live model discovery
- Parameters: duration, resolution, aspect ratio, audio, seed
- Frame images (image-to-video: first/last frame)
- Cost estimation based on parsed pricing data
- Generated video viewer + download + history
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from datetime import datetime
from io import BytesIO
from typing import Any

import requests
import streamlit as st
import yaml
from PIL import Image

from src.llm_lite import llm_inference
from src.openrouter_video_universal import (
    OpenRouterVideoClient,
    VideoRequest,
    make_frame_image,
    make_input_reference,
)
from utils import render_image_selector

# ---------------------------------------------------------------------------
# VIDEO MODEL CATALOG
# ---------------------------------------------------------------------------
OPENROUTER_VIDEO_MODELS = {
    # Google Veo
    "veo-3.1":             "google/veo-3.1",
    "veo-3.1-fast":        "google/veo-3.1-fast",
    "veo-3.1-lite":        "google/veo-3.1-lite",
    # ByteDance Seedance
    "seedance-1.5-pro":    "bytedance/seedance-1-5-pro",
    "seedance-2.0":        "bytedance/seedance-2.0",
    "seedance-2.0-fast":   "bytedance/seedance-2.0-fast",
    # Kuaishou Kling
    "kling-v3-pro":        "kwaivgi/kling-v3.0-pro",
    "kling-v3-std":        "kwaivgi/kling-v3.0-std",
    "kling-o1":            "kwaivgi/kling-video-o1",
    # Alibaba Wan
    "wan-2.7":             "alibaba/wan-2.7",
    "wan-2.6":             "alibaba/wan-2.6",
    # MiniMax
    "hailuo-2.3":          "minimax/hailuo-2.3",
    # OpenAI
    "sora-2-pro":          "openai/sora-2-pro",
    # xAI
    "grok-imagine-video":  "x-ai/grok-imagine-video",
}

# ---------------------------------------------------------------------------
# COST TABLE  (cost per second, per resolution, with/without audio)
# Structure: {model_id: {resolution: {audio: price, no_audio: price}}}
# None values mean unknown pricing for that combination.
# ---------------------------------------------------------------------------
VIDEO_COSTS: dict[str, Any] = {
    "google/veo-3.1": {
        "1080p": {"audio": 0.40, "no_audio": 0.20},
        "4K":    {"audio": 0.60, "no_audio": 0.40},
        "_default": {"audio": 0.40, "no_audio": 0.20},
    },
    "google/veo-3.1-lite": {
        "720p":  {"audio": 0.05, "no_audio": 0.03},
        "1080p": {"audio": 0.08, "no_audio": 0.05},
        "_default": {"audio": 0.05, "no_audio": 0.03},
    },
    "google/veo-3.1-fast": {
        "720p":  {"audio": 0.10, "no_audio": 0.08},
        "1080p": {"audio": 0.12, "no_audio": 0.10},
        "4K":    {"audio": 0.30, "no_audio": 0.25},
        "_default": {"audio": 0.10, "no_audio": 0.08},
    },
    "openai/sora-2-pro": {
        "720p":  {"audio": 0.30, "no_audio": 0.30},
        "1080p": {"audio": 0.50, "no_audio": 0.50},
        "_default": {"audio": 0.30, "no_audio": 0.30},
    },
    "bytedance/seedance-1-5-pro": {
        "480p":  {"audio": 0.02306, "no_audio": 0.01153},
        "720p":  {"audio": 0.05184, "no_audio": 0.02592},
        "1080p": {"audio": 0.11660, "no_audio": 0.05832},
        "_default": {"audio": 0.05184, "no_audio": 0.02592},
    },
    "bytedance/seedance-2.0": {
        "480p":  {"audio": 0.06726, "no_audio": 0.06726},
        "720p":  {"audio": 0.15120, "no_audio": 0.15120},
        "1080p": {"audio": 0.34020, "no_audio": 0.34020},
        "_default": {"audio": 0.15120, "no_audio": 0.15120},
    },
    "bytedance/seedance-2.0-fast": {
        "480p":  {"audio": 0.05380, "no_audio": 0.05380},
        "720p":  {"audio": 0.12100, "no_audio": 0.12100},
        "1080p": {"audio": 0.27220, "no_audio": 0.27220},
        "_default": {"audio": 0.12100, "no_audio": 0.12100},
    },
    "alibaba/wan-2.6": {
        "480p":  {"audio": 0.04, "no_audio": 0.04},
        "720p":  {"audio": 0.08, "no_audio": 0.08},
        "1080p": {"audio": 0.12, "no_audio": 0.12},
        "_default": {"audio": 0.08, "no_audio": 0.08},
    },
    "alibaba/wan-2.7": {
        "_default": {"audio": 0.10, "no_audio": 0.10},
    },
    "minimax/hailuo-2.3": {
        "_default": {"audio": 0.0817, "no_audio": 0.0817},
    },
    "kwaivgi/kling-video-o1": {
        "_default": {"audio": 0.112, "no_audio": 0.112},
    },
    "kwaivgi/kling-v3.0-std": {
        "_default": {"audio": 0.126, "no_audio": 0.084},
    },
    "kwaivgi/kling-v3.0-pro": {
        "_default": {"audio": 0.168, "no_audio": 0.112},
    },
    "x-ai/grok-imagine-video": {
        "480p": {"audio": 0.05, "no_audio": 0.05},
        "720p": {"audio": 0.07, "no_audio": 0.07},
        "_default": {"audio": 0.05, "no_audio": 0.05},
    },
}

VIDEO_RESOLUTIONS = ["480p", "720p", "1080p", "2K", "4K"]
VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"]
VIDEO_DURATIONS = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]

LLM_VIDEO_ENHANCE_MODEL = "x-ai/grok-4.3"

# ---------------------------------------------------------------------------
# VIDEO-SPECIFIC LLM SYSTEM PROMPTS
# ---------------------------------------------------------------------------
VIDEO_LLM_ENHANCE_SYSTEM = (
    "You are an expert video-prompt engineer for text-to-video AI models. "
    "Lightly enhance the user's prompt by adding modest cinematic detail: "
    "camera movement (pan, dolly, zoom), lighting quality, temporal transitions, "
    "motion description, and visual style — ONLY where the prompt is sparse. "
    "Preserve the original subject, intent, structure, and length. "
    "If the prompt is already detailed, return it nearly unchanged. "
    "Do NOT invent new subjects or scene elements. Keep the original language. "
    "Return ONLY the enhanced prompt as plain text, no preamble, no quotes, no explanation."
)

VIDEO_LLM_FIX_SYSTEM = (
    "You are a proofreader for video generation prompts. Fix ONLY grammar, spelling, "
    "punctuation, and syntax errors. Do NOT add new content. Do NOT change style, tone, "
    "vocabulary, or meaning. Preserve technical tags and artist names verbatim. "
    "If there are no errors, return the prompt unchanged. "
    "Return ONLY the corrected prompt as plain text, no preamble, no quotes, no explanation."
)

VIDEO_LLM_RESTYLE_SYSTEM = (
    "You are an expert video-prompt editor performing a CONSERVATIVE RESTYLE. "
    "You will receive an input formatted as:\n"
    "ORIGINAL PROMPT:\n<the prompt to restyle>\n\n"
    "USER INSTRUCTIONS:\n<short restyle directions, possibly in any language>\n\n"
    "Your task: rewrite the ORIGINAL PROMPT applying ONLY the changes implied by the "
    "USER INSTRUCTIONS. The user's instructions may be in ANY language — understand them "
    "regardless of language, but ALWAYS write the output in English.\n\n"
    "Strict rules:\n"
    "- Be CONSERVATIVE: preserve subject, motion, camera angle, setting, characters.\n"
    "- Modify ONLY the aspects the user explicitly asks to change (style, mood, lighting, "
    "camera movement, era, pacing). Leave everything else untouched.\n"
    "- Do NOT add new subjects, objects, characters, or scene elements not requested.\n"
    "- Keep the original prompt's structure and length as close as possible.\n"
    "- If the user instructions are empty or unclear, return the ORIGINAL PROMPT unchanged.\n\n"
    "Return ONLY the restyled prompt as plain text — no preamble, no quotes, no explanation."
)


# ---------------------------------------------------------------------------
# COST ESTIMATION
# ---------------------------------------------------------------------------
def estimate_cost(model_id: str, duration: int, resolution: str, generate_audio: bool) -> str | None:
    """Return a human-readable cost estimate string, or None if unknown."""
    costs = VIDEO_COSTS.get(model_id)
    if not costs:
        return None
    res_costs = costs.get(resolution) or costs.get("_default")
    if not res_costs:
        return None
    key = "audio" if generate_audio else "no_audio"
    price_per_sec = res_costs.get(key)
    if price_per_sec is None:
        return None
    total = price_per_sec * duration
    return f"≈ ${total:.4f}  (${price_per_sec:.4f}/s × {duration}s)"


# ---------------------------------------------------------------------------
# PROMPT ENHANCER (video-specific, self-contained)
# ---------------------------------------------------------------------------
def render_video_prompt_enhancer(prompt: str, session_key: str = "video_llm_enhanced_prompt") -> str:
    """Video-specific prompt enhance/fix/restyle widget. Returns effective prompt."""

    def _run_llm(system_prompt: str, label: str, user_payload: str | None = None):
        src = (prompt or "").strip()
        if not src:
            st.warning("⚠️ Prompt is empty.")
            return
        try:
            with st.spinner(f"{label} via {LLM_VIDEO_ENHANCE_MODEL}..."):
                result = llm_inference(
                    prompt=user_payload if user_payload is not None else src,
                    system=system_prompt,
                    model=LLM_VIDEO_ENHANCE_MODEL,
                    temperature=0.7,
                )
            if result and result.strip():
                st.session_state[session_key] = result.strip()
                st.rerun()
            else:
                st.warning("⚠️ LLM returned empty response.")
        except Exception as e:
            st.error(f"❌ LLM {label.lower()} failed: {e}")

    restyle_key = f"{session_key}_restyle_instructions"
    restyle_instructions = st.text_input(
        "🎬 Restyle instructions (any language)",
        key=restyle_key,
        placeholder="e.g. slow motion cinematic / stile anime / camera drone shot",
        help="Short directions for the Restyle button. The original prompt is kept "
             "intact except for the aspects you mention here.",
    )

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button("✨ Enhance", key=f"{session_key}_btn_enhance",
                     help=f"Add cinematic detail via {LLM_VIDEO_ENHANCE_MODEL}"):
            _run_llm(VIDEO_LLM_ENHANCE_SYSTEM, "Enhancing")
    with cols[1]:
        if st.button("🛠️ Fix", key=f"{session_key}_btn_fix",
                     help="Fix grammar/syntax errors only"):
            _run_llm(VIDEO_LLM_FIX_SYSTEM, "Fixing")
    with cols[2]:
        if st.button("🎬 Restyle", key=f"{session_key}_btn_restyle",
                     help="Conservative restyle using only your instructions above"):
            instr = (restyle_instructions or "").strip()
            if not instr:
                st.warning("⚠️ Enter restyle instructions first.")
            else:
                payload = (
                    f"ORIGINAL PROMPT:\n{(prompt or '').strip()}\n\n"
                    f"USER INSTRUCTIONS:\n{instr}"
                )
                _run_llm(VIDEO_LLM_RESTYLE_SYSTEM, "Restyling", user_payload=payload)
    with cols[3]:
        if st.session_state.get(session_key) and st.button(
            "🗑️ Clear", key=f"{session_key}_btn_clear"
        ):
            st.session_state.pop(session_key, None)
            st.rerun()

    if st.session_state.get(session_key):
        edited = st.text_area(
            "✨ Enhanced Prompt (editable — used for generation)",
            value=st.session_state[session_key],
            height=200,
            key=f"{session_key}_editor",
            help="Edit freely. This text overrides the prompt above."
        )
        st.session_state[session_key] = edited
        return edited

    return prompt


# ---------------------------------------------------------------------------
# PROMPT LOADING (reused from app.py logic)
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    if st.session_state.get("api_key"):
        return st.session_state.api_key
    if os.path.exists("api_keys.json"):
        try:
            with open("api_keys.json", "r") as f:
                return json.load(f).get("openrouter", "")
        except Exception:
            pass
    return ""


def _load_yaml_prompts(file_path: str) -> dict:
    full = os.path.join("prompts", file_path)
    if not os.path.exists(full):
        return {}
    try:
        with open(full, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        st.error(f"Error loading YAML: {e}")
        return {}


def _load_json_prompts(file_path: str) -> dict:
    full = os.path.join("prompts", file_path)
    if not os.path.exists(full):
        return {}
    try:
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return {}


def _flatten_yaml_prompts(data: dict) -> list[dict]:
    flattened = []

    def recurse(d, path="", section=""):
        for key, value in d.items():
            current_path = f"{path} > {key}" if path else key
            if isinstance(value, list):
                for i, p in enumerate(value):
                    if isinstance(p, str) and not p.strip().startswith("#"):
                        flattened.append({"section": section, "category": current_path, "prompt": p.strip(), "index": i})
            elif isinstance(value, dict):
                recurse(value, current_path, section)

    if "create_prompts" in data:
        recurse(data["create_prompts"], "", "Create")
    if "edit_prompts" in data:
        recurse(data["edit_prompts"], "", "Edit")
    return flattened


def _flatten_json_prompts(data: dict) -> list[dict]:
    flattened = []
    if not isinstance(data, dict):
        return flattened

    def dump_p(v):
        return json.dumps(v, indent=2, ensure_ascii=False)

    processed = set()

    def process_section(key, label):
        section_data = data.get(key)
        if not isinstance(section_data, dict):
            return
        processed.add(key)
        for cat_key, cat_val in section_data.items():
            if isinstance(cat_val, dict) and cat_val:
                for pname, pval in cat_val.items():
                    flattened.append({"section": label, "category": cat_key, "prompt_name": pname, "prompt": dump_p(pval), "index": 0})
            else:
                flattened.append({"section": label, "category": cat_key, "prompt_name": cat_key, "prompt": dump_p(cat_val), "index": 0})

    process_section("Create_Prompts", "Create")
    process_section("create_prompts", "Create")
    process_section("Edit_Prompts", "Edit")
    process_section("edit_prompts", "Edit")

    for k, v in data.items():
        if k in processed or k.lower() in ["source", "sources"]:
            continue
        flattened.append({"section": "JSON Prompts", "category": k, "prompt_name": k, "prompt": dump_p(v), "index": 0})

    return flattened


# ---------------------------------------------------------------------------
# ASPECT RATIO DETECTION
# ---------------------------------------------------------------------------
def _get_image_aspect_ratio(image: Image.Image) -> str:
    """Return the closest standard video aspect ratio for the given image."""
    w, h = image.size
    ratio = w / h
    candidates = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "1:1":  1.0,
        "4:3":  4 / 3,
        "3:4":  3 / 4,
        "3:2":  3 / 2,
        "2:3":  2 / 3,
        "21:9": 21 / 9,
        "9:21": 9 / 21,
    }
    return min(candidates, key=lambda k: abs(candidates[k] - ratio))


# ---------------------------------------------------------------------------
# IMAGE → BASE64 URL (for frame images)
# ---------------------------------------------------------------------------
def _pil_to_data_url(img: Image.Image, max_size: int = 1024) -> str:
    w, h = img.size
    if max(w, h) > max_size:
        if w >= h:
            img = img.resize((max_size, int(h * max_size / w)), Image.Resampling.LANCZOS)
        else:
            img = img.resize((int(w * max_size / h), max_size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# VIDEO SAVING
# ---------------------------------------------------------------------------
def _save_video(data: bytes, prompt: str, model_name: str, output_folder: str = "outputs/videos") -> str:
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_short = "".join(c for c in prompt[:30] if c.isalnum() or c in ("_", "-")).replace(" ", "_")
    model_short = model_name.split("/")[-1]
    filename = os.path.join(output_folder, f"{prompt_short}_{model_short}_{timestamp}.mp4")
    with open(filename, "wb") as f:
        f.write(data)
    return filename


JOBS_LOG_FILE = "batch_jobs_log.json"


def _persist_job(job_id: str, polling_url: str, req: VideoRequest):
    """Append job metadata to batch_jobs_log.json so it survives session restarts."""
    entry = {
        "job_id": job_id,
        "polling_url": polling_url,
        "model": req.model,
        "prompt": req.prompt,
        "duration": req.duration,
        "resolution": req.resolution,
        "aspect_ratio": req.aspect_ratio,
        "generate_audio": req.generate_audio,
        "seed": req.seed,
        "submitted_at": datetime.now().isoformat(),
        "status": "submitted",
    }
    log: list = []
    if os.path.exists(JOBS_LOG_FILE):
        try:
            with open(JOBS_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    # avoid duplicates
    if not any(j.get("job_id") == job_id for j in log):
        log.append(entry)
    with open(JOBS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def _update_persisted_job(job_id: str, status: str, saved_path: str | None = None):
    """Update status of a job already in batch_jobs_log.json."""
    if not os.path.exists(JOBS_LOG_FILE):
        return
    try:
        with open(JOBS_LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
        for entry in log:
            if entry.get("job_id") == job_id:
                entry["status"] = status
                if saved_path:
                    entry["saved_path"] = saved_path
        with open(JOBS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _load_persisted_jobs() -> list[dict]:
    if not os.path.exists(JOBS_LOG_FILE):
        return []
    try:
        with open(JOBS_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LOG HELPER
# ---------------------------------------------------------------------------
def _log_event(event: str, data: Any = None):
    entry = {"_ts": datetime.now().strftime("%H:%M:%S"), "event": event, "data": data}
    log: list = st.session_state.get("video_gen_log") or []
    log.append(entry)
    st.session_state["video_gen_log"] = log


def _safe_payload_for_log(payload: dict) -> dict:
    """Truncate base64 blobs so the log stays readable."""
    safe = json.loads(json.dumps(payload))
    for key in ("frame_images", "input_references"):
        for item in safe.get(key, []):
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                item["image_url"]["url"] = url[:60] + "…[base64 truncated]"
    return safe


# ---------------------------------------------------------------------------
# SUBMIT (called once on button press — synchronous, fast)
# ---------------------------------------------------------------------------
def _submit_job(api_key: str, req: VideoRequest):
    """Submit the job to OpenRouter and store job_id + polling_url in session_state."""
    _log_event("submit_payload", _safe_payload_for_log(req.to_payload()))
    try:
        client = OpenRouterVideoClient(api_key=api_key)
        sub = client.submit(req)
    except Exception as e:
        _log_event("submit_error", str(e))
        st.session_state.video_gen_error = f"Submit error: {e}"
        st.session_state.video_gen_status = "error"
        st.session_state.video_gen_running = False
        return

    _log_event("submit_response", sub)

    polling_url = sub.get("polling_url", "")
    if not polling_url:
        err = sub.get("error") or sub.get("message") or "No polling_url in response"
        st.session_state.video_gen_error = f"API error: {err}"
        st.session_state.video_gen_status = "error"
        st.session_state.video_gen_running = False
        return

    job_id = sub.get("id", "?")
    st.session_state.video_gen_job_id = job_id
    st.session_state.video_gen_polling_url = polling_url
    st.session_state.video_gen_status = f"Job {job_id} submitted — waiting..."
    st.session_state.video_gen_poll_count = 0

    # Persist job to disk so it can be recovered across sessions
    _persist_job(job_id, polling_url, req)


def _do_poll(api_key: str):
    """Poll the current job once and update session_state. Called on button press."""
    polling_url: str = st.session_state.get("video_gen_polling_url", "")
    if not polling_url:
        st.session_state.video_gen_error = "No polling URL — submit a job first."
        return

    poll_count = st.session_state.get("video_gen_poll_count", 0) + 1
    st.session_state.video_gen_poll_count = poll_count

    try:
        client = OpenRouterVideoClient(api_key=api_key)
        status_resp = client.poll(polling_url)
    except Exception as e:
        _log_event(f"poll_error_{poll_count}", str(e))
        st.session_state.video_gen_error = f"Polling error: {e}"
        st.session_state.video_gen_status = "error"
        st.session_state.video_gen_running = False
        return

    s = status_resp.get("status")
    _log_event(f"poll_{poll_count}", status_resp)
    st.session_state.video_gen_status = f"{s}"

    terminal = {"completed", "failed", "cancelled", "expired"}
    if s not in terminal:
        return  # still processing — user will press the button again when ready

    # Job reached a terminal state — re-enable the Generate button
    st.session_state.video_gen_running = False

    if s == "completed":
        urls = status_resp.get("unsigned_urls") or []
        if urls:
            video_url = urls[0]
            if video_url.startswith("/"):
                video_url = f"https://openrouter.ai{video_url}"
            _log_event("download_start", {"url": video_url})
            try:
                resp = requests.get(
                    video_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()
                video_bytes = resp.content
                _log_event("download_ok", {"bytes": len(video_bytes)})
                st.session_state.video_gen_result = video_bytes
                st.session_state.video_gen_status = "completed"

                # Auto-save to outputs/videos exactly once, right after download
                saved_path = None
                if st.session_state.get("video_auto_save", True):
                    try:
                        saved_path = _save_video(
                            video_bytes,
                            st.session_state.get("video_gen_prompt_used", ""),
                            st.session_state.get("video_gen_model_used", "video"),
                        )
                        _log_event("auto_saved", {"path": saved_path})
                    except Exception as e:
                        _log_event("auto_save_error", str(e))
                st.session_state.video_gen_saved_path = saved_path

                _update_persisted_job(
                    st.session_state.get("video_gen_job_id", "?"), "completed", saved_path
                )
            except Exception as e:
                _log_event("download_error", str(e))
                st.session_state.video_gen_error = f"Download error: {e}"
                st.session_state.video_gen_status = "error"
        else:
            _log_event("no_urls", status_resp)
            st.session_state.video_gen_error = "Completed but no video URL returned."
            st.session_state.video_gen_status = "error"
    else:
        err = status_resp.get("error") or status_resp.get("message") or s
        _log_event("job_failed", status_resp)
        st.session_state.video_gen_error = f"Job {s}: {err}"
        st.session_state.video_gen_status = "error"
        _update_persisted_job(
            st.session_state.get("video_gen_job_id", "?"), s
        )


# ---------------------------------------------------------------------------
# LAST-FRAME EXTRACTION
# ---------------------------------------------------------------------------
def _extract_last_frame(video_bytes: bytes) -> Image.Image:
    """Decode a video and return its last frame as a PIL Image (RGB)."""
    import tempfile

    import cv2  # lazy import — only needed by this feature

    # OpenCV needs a real file path, not raw bytes
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the video (unsupported codec/format).")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_bgr = None
        if total > 0:
            # Seek near the end, then read forward to the actual last decodable frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(total - 5, 0))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last_bgr = frame
        if last_bgr is None:
            # Fallback: frame count unknown or seek failed — scan the whole stream
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last_bgr = frame
        cap.release()

        if last_bgr is None:
            raise RuntimeError("No decodable frames found in the video.")

        rgb = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _last_frame_extractor_box():
    """UI box: upload a video and extract + save its last frame as an image."""
    st.subheader("🖼️ Extract Last Frame from Video")
    st.caption(
        "Upload a video and grab its last frame — handy as a `last_frame` / "
        "reference image or to continue a clip in a new generation."
    )

    up = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "webm", "mkv", "avi"],
        key="last_frame_video_upload",
    )
    if up is None:
        return

    if st.button("🎞️ Extract last frame", key="extract_last_frame_btn"):
        with st.spinner("Decoding video…"):
            try:
                frame = _extract_last_frame(up.getvalue())
            except Exception as e:
                st.error(f"❌ Could not extract frame: {e}")
                return

        os.makedirs("outputs/frames", exist_ok=True)
        stem = os.path.splitext(os.path.basename(up.name))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{stem}_lastframe_{ts}.png"
        fpath = os.path.join("outputs/frames", fname)
        frame.save(fpath)

        # Keep result across reruns so the download button works
        buf = BytesIO()
        frame.save(buf, format="PNG")
        st.session_state["last_frame_result"] = {
            "png": buf.getvalue(),
            "path": fpath,
            "name": fname,
            "size": frame.size,
        }

    res = st.session_state.get("last_frame_result")
    if res:
        st.image(res["png"], caption=f"Last frame — {res['size'][0]}×{res['size'][1]}")
        st.success(f"Saved to `{res['path']}`")
        st.download_button(
            "⬇️ Download last frame (PNG)",
            data=res["png"],
            file_name=res["name"],
            mime="image/png",
            key="last_frame_download_btn",
        )


# ---------------------------------------------------------------------------
# VIDEO MERGING / CONCATENATION
# ---------------------------------------------------------------------------
def _ffmpeg_exe() -> str:
    """Return the ffmpeg executable, preferring imageio's bundled binary."""
    from shutil import which

    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # last resort — let it fail with a clear error


def _probe_resolution(path: str) -> tuple[int, int]:
    """Read (width, height) of the first video stream via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Could not read resolution of {os.path.basename(path)}")
    return w, h


def _merge_videos(
    video_files: list[tuple[str, bytes]],
    out_path: str,
    *,
    fps: int = 30,
) -> str:
    """
    Concatenate videos back-to-back (no gaps: last frame of clip N is immediately
    followed by the first frame of clip N+1) into a single MP4.

    Clips may come from different models with different sizes/fps/codecs, so every
    input is re-encoded and normalized to a common resolution (that of the first
    clip), fps and pixel format. Missing audio tracks are padded with silence so
    the concat filter always gets matching stream layouts.
    """
    import subprocess
    import tempfile

    if len(video_files) < 2:
        raise ValueError("Need at least 2 videos to merge.")

    tmp_dir = tempfile.mkdtemp(prefix="vidmerge_")
    tmp_paths: list[str] = []
    try:
        for i, (name, data) in enumerate(video_files):
            ext = os.path.splitext(name)[1] or ".mp4"
            p = os.path.join(tmp_dir, f"in_{i}{ext}")
            with open(p, "wb") as f:
                f.write(data)
            tmp_paths.append(p)

        # Target resolution = first clip; force even dimensions (H.264 requirement)
        tw, th = _probe_resolution(tmp_paths[0])
        tw -= tw % 2
        th -= th % 2

        ffmpeg = _ffmpeg_exe()
        cmd: list[str] = [ffmpeg, "-y"]
        for p in tmp_paths:
            cmd += ["-i", p]

        # Build filter_complex: normalize each input, then concat v+a
        n = len(tmp_paths)
        parts: list[str] = []
        concat_inputs = ""
        for i in range(n):
            # scale keeping AR, pad to target, unify SAR/fps; ensure an audio track
            parts.append(
                f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}];"
                f"[{i}:a?]aresample=async=1[a{i}pre];"
                f"anullsrc=channel_layout=stereo:sample_rate=48000[a{i}sil];"
                f"[a{i}pre][a{i}sil]amix=inputs=2:duration=first:dropout_transition=0[a{i}]"
            )
            concat_inputs += f"[v{i}][a{i}]"
        filter_complex = ";".join(parts) + (
            f";{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"
        )

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise RuntimeError(f"ffmpeg failed:\n{tail}")
        return out_path
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _video_merger_box():
    """UI box: upload multiple videos and merge them into one continuous clip."""
    st.subheader("🎬 Merge Videos (Sequential Concat)")
    st.caption(
        "Upload two or more videos to join them back-to-back into one clip — "
        "the last frame of each is directly followed by the first frame of the "
        "next, no gaps. Clips are normalized to the first video's resolution, so "
        "you can mix outputs from different models."
    )

    ups = st.file_uploader(
        "Upload videos (order = concat order)",
        type=["mp4", "mov", "webm", "mkv", "avi"],
        accept_multiple_files=True,
        key="merge_videos_upload",
    )
    if not ups:
        return

    st.markdown("**Merge order:**")
    for i, f in enumerate(ups, 1):
        st.markdown(f"{i}. `{f.name}`")
    st.caption("Tip: to reorder, remove the files and re-upload them in the order you want.")

    fps = st.number_input(
        "Output FPS", min_value=8, max_value=60, value=30, step=1, key="merge_fps"
    )

    if len(ups) < 2:
        st.info("Upload at least 2 videos to merge.")
        return

    if st.button("🔗 Merge videos", key="merge_videos_btn", type="primary"):
        os.makedirs("outputs/videos", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"merged_{len(ups)}clips_{ts}.mp4"
        fpath = os.path.join("outputs/videos", fname)
        video_files = [(f.name, f.getvalue()) for f in ups]

        with st.spinner(f"Merging {len(ups)} videos… (re-encoding, this can take a bit)"):
            try:
                _merge_videos(video_files, fpath, fps=int(fps))
            except Exception as e:
                st.error(f"❌ Merge failed: {e}")
                return

        with open(fpath, "rb") as fv:
            merged_bytes = fv.read()
        st.session_state["merged_video_result"] = {
            "bytes": merged_bytes,
            "path": fpath,
            "name": fname,
        }

    res = st.session_state.get("merged_video_result")
    if res:
        st.success(f"✅ Merged & saved to `{res['path']}`")
        st.video(res["bytes"])
        st.download_button(
            "⬇️ Download merged video (MP4)",
            data=res["bytes"],
            file_name=res["name"],
            mime="video/mp4",
            key="merged_video_download_btn",
        )


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------
def show_video_generator_page():
    col_title, col_recover = st.columns([3, 1])
    with col_title:
        st.title("🎬 OpenRouter Video Generator")
        st.markdown("Generate videos using OpenRouter video models with full parameter control.")
    with col_recover:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical spacer
        if st.button("🔄 Recover Past Jobs", key="video_recover_shortcut", use_container_width=True,
                     help="List all past jobs from your OpenRouter account and download completed videos"):
            st.session_state["video_scroll_to_recover"] = True

    # ── Session state init ──────────────────────────────────────────────────
    for key, default in [
        ("video_generated_history", []),
        ("video_gen_status", None),
        ("video_gen_result", None),
        ("video_gen_error", None),
        ("video_gen_running", False),
        ("video_gen_prompt_used", ""),
        ("video_gen_model_used", ""),
        ("video_gen_params_used", {}),
        ("video_gen_log", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    api_key = _load_api_key()

    # ── Sidebar: Video parameters ───────────────────────────────────────────
    with st.sidebar:
        st.subheader("🎬 Video Settings")

        selected_model_key = st.selectbox(
            "Model",
            options=list(OPENROUTER_VIDEO_MODELS.keys()),
            index=0,
            help="Choose the video generation model",
            key="video_model_select",
        )
        model_id = OPENROUTER_VIDEO_MODELS[selected_model_key]

        st.divider()
        st.subheader("⏱️ Duration & Resolution")

        duration = st.select_slider(
            "Duration (seconds)",
            options=VIDEO_DURATIONS,
            value=5,
            key="video_duration",
        )

        resolution = st.selectbox(
            "Resolution",
            options=VIDEO_RESOLUTIONS,
            index=1,  # 720p default
            key="video_resolution",
        )

        use_auto_aspect = st.checkbox(
            "Auto-detect aspect ratio from frame image",
            value=True,
            key="video_use_auto_aspect",
            help="Automatically use the aspect ratio of the first frame image uploaded",
        )

        aspect_ratio_manual = st.selectbox(
            "Aspect Ratio",
            options=VIDEO_ASPECT_RATIOS,
            index=0,  # 16:9 default
            disabled=use_auto_aspect,
            key="video_aspect_ratio",
            help="Target aspect ratio for generated video",
        )
        aspect_ratio = aspect_ratio_manual

        generate_audio = st.checkbox(
            "🔊 Generate Audio",
            value=True,
            key="video_generate_audio",
            help="Enable audio generation (supported models only)",
        )

        st.divider()
        st.subheader("🎲 Seed")
        use_random_seed = st.checkbox("Use random seed", value=True, key="video_random_seed")
        seed = None
        if not use_random_seed:
            seed = st.number_input("Seed", min_value=1, max_value=1_000_000, value=42, key="video_seed")

        st.divider()
        st.subheader("💾 Output Settings")
        auto_save_video = st.checkbox("Auto-save generated videos", value=True, key="video_auto_save")

        st.divider()

        # ── Cost preview ────────────────────────────────────────────────────
        st.subheader("💰 Cost Estimate")
        cost_str = estimate_cost(model_id, duration, resolution, generate_audio)
        if cost_str:
            st.info(cost_str)
        else:
            st.caption("Cost data not available for this model/config.")

        # ── Prompt source ───────────────────────────────────────────────────
        st.divider()
        st.subheader("📂 Prompt Source")
        PROMPTS_FILES = {"base": "prompts", "custom": "prompts_custom", "assets": "prompts_assets"}
        default_prompts = (
            "custom"
            if (os.path.exists("prompts/prompts_custom.json") or os.path.exists("prompts/prompts_custom.yaml"))
            else "base"
        )
        selected_prompts = st.selectbox(
            "Prompts File",
            options=list(PROMPTS_FILES.keys()),
            index=list(PROMPTS_FILES.keys()).index(default_prompts),
            key="video_prompts_source",
        )

    # ── Main area ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    # ─────────────────────────────────────────────────────── LEFT COLUMN ───
    with col1:
        st.subheader("📝 Prompt & Frame Images")

        prompt_source = st.radio(
            "Prompt Source",
            options=["Custom Prompt", "Load from YAML", "Load from JSON"],
            horizontal=True,
            key="video_prompt_source_radio",
        )

        prompt = ""

        # ── JSON prompt loader ──────────────────────────────────────────────
        if prompt_source == "Load from JSON":
            json_file = PROMPTS_FILES[selected_prompts] + ".json"
            json_data = _load_json_prompts(json_file)
            flat_json = _flatten_json_prompts(json_data)

            if flat_json:
                sections: dict[str, list] = {}
                for item in flat_json:
                    s = item.get("section", "JSON Prompts") or "JSON Prompts"
                    sections.setdefault(s, []).append(item)

                c_sec, c_cat = st.columns([1, 2])
                with c_sec:
                    selected_section = st.selectbox("Section", list(sections.keys()), key="vg_json_section")
                section_items = sections[selected_section]

                with c_cat:
                    cats: dict[str, list] = {}
                    for item in section_items:
                        c = item.get("category", "Uncategorized")
                        cats.setdefault(c, []).append(item)
                    selected_cat = st.selectbox("Category", list(cats.keys()), key="vg_json_cat")

                cat_items = cats[selected_cat]
                prompt_names = [i.get("prompt_name", f"Prompt {j}") for j, i in enumerate(cat_items)]
                selected_pname = st.selectbox("Prompt", prompt_names, key="vg_json_prompt")
                selected_item = cat_items[prompt_names.index(selected_pname)]
                prompt = selected_item.get("prompt", "")
            else:
                st.info(f"No prompts found in {json_file}")

        # ── YAML prompt loader ──────────────────────────────────────────────
        elif prompt_source == "Load from YAML":
            yaml_file = PROMPTS_FILES[selected_prompts] + ".yaml"
            yaml_data = _load_yaml_prompts(yaml_file)
            flat_yaml = _flatten_yaml_prompts(yaml_data)

            if flat_yaml:
                sections_y: dict[str, list] = {}
                for item in flat_yaml:
                    s = item.get("section", "Prompts") or "Prompts"
                    sections_y.setdefault(s, []).append(item)

                cy_sec, cy_cat = st.columns([1, 2])
                with cy_sec:
                    selected_section_y = st.selectbox("Section", list(sections_y.keys()), key="vg_yaml_section")
                section_items_y = sections_y[selected_section_y]

                with cy_cat:
                    cats_y: dict[str, list] = {}
                    for item in section_items_y:
                        c = item.get("category", "Uncategorized")
                        cats_y.setdefault(c, []).append(item)
                    selected_cat_y = st.selectbox("Category", list(cats_y.keys()), key="vg_yaml_cat")

                cat_items_y = cats_y[selected_cat_y]
                prompt_idx = st.selectbox(
                    "Prompt",
                    options=range(len(cat_items_y)),
                    format_func=lambda i: cat_items_y[i]["prompt"][:80] + "..." if len(cat_items_y[i]["prompt"]) > 80 else cat_items_y[i]["prompt"],
                    key="vg_yaml_prompt_idx",
                )
                prompt = cat_items_y[prompt_idx]["prompt"]
            else:
                st.info(f"No prompts found in {yaml_file}")

        # ── Custom prompt textarea ──────────────────────────────────────────
        prompt_text = st.text_area(
            "Video Prompt",
            value=prompt,
            height=150,
            key="video_prompt_textarea",
            placeholder="Describe the video you want to generate. Be specific about motion, camera movement, lighting, and scene details.",
        )
        prompt = prompt_text

        # ── Video Prompt Enhancer ───────────────────────────────────────────
        with st.expander("✨ Prompt Enhancer (LLM)", expanded=False):
            prompt = render_video_prompt_enhancer(prompt, session_key="video_llm_enhanced_prompt")

        st.divider()

        # ── Frame Images (image-to-video) ───────────────────────────────────
        st.subheader("🖼️ Frame Images (Image-to-Video)")
        st.caption("Upload images to use as first/last frame for image-to-video generation. Not all models support this.")

        frame_mode = st.radio(
            "Frame image mode",
            options=["None", "First Frame", "Last Frame", "First + Last Frame"],
            horizontal=True,
            key="video_frame_mode",
        )

        first_frame_img = None
        last_frame_img = None

        if frame_mode in ("First Frame", "First + Last Frame"):
            uploaded_first = st.file_uploader("First Frame Image", type=["png", "jpg", "jpeg", "webp"], key="video_first_frame")
            if uploaded_first:
                first_frame_img = Image.open(uploaded_first)
                st.image(first_frame_img, caption="First Frame", use_container_width=True)

        if frame_mode in ("Last Frame", "First + Last Frame"):
            uploaded_last = st.file_uploader("Last Frame Image", type=["png", "jpg", "jpeg", "webp"], key="video_last_frame")
            if uploaded_last:
                last_frame_img = Image.open(uploaded_last)
                st.image(last_frame_img, caption="Last Frame", use_container_width=True)

        # ── Reference images (reference-to-video) ──────────────────────────
        st.subheader("📌 Reference Images (Style / Character)")
        st.caption("Upload reference images for style/character guidance (reference-to-video). Not all models support this.")
        ref_upload = st.file_uploader(
            "Reference Images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="video_ref_images",
        )
        ref_images_pil: list[Image.Image] = []
        if ref_upload:
            for f in ref_upload:
                img = Image.open(f)
                ref_images_pil.append(img)
            if ref_images_pil:
                cols_ref = st.columns(min(len(ref_images_pil), 3))
                for i, ri in enumerate(ref_images_pil[:3]):
                    cols_ref[i].image(ri, use_container_width=True)

        # ── Folder image selector ───────────────────────────────────────────
        with st.expander("📁 Select frame/reference from folder", expanded=False):
            folder_imgs = render_image_selector(session_key="video_folder_sel")
            if folder_imgs:
                st.caption(f"{len(folder_imgs)} image(s) selected from folder. They will be used as reference images.")
                ref_images_pil = folder_imgs + ref_images_pil

    # ─────────────────────────────────────────────────────── RIGHT COLUMN ──
    with col2:
        st.subheader("🎬 Generate & Preview")

        # ── Generate button ─────────────────────────────────────────────────
        can_generate = bool(prompt and prompt.strip()) and bool(api_key)
        if not api_key:
            st.warning("⚠️ No API key. Set your OpenRouter API key in the sidebar.")

        gen_btn = st.button(
            "🚀 Generate Video",
            disabled=not can_generate or st.session_state.video_gen_running,
            key="video_generate_btn",
            use_container_width=True,
            type="primary",
        )

        if gen_btn and can_generate:
            # Build frame_images list
            frame_images_payload: list[dict] = []
            if first_frame_img:
                url = _pil_to_data_url(first_frame_img)
                frame_images_payload.append(make_frame_image(url, "first_frame"))
            if last_frame_img:
                url = _pil_to_data_url(last_frame_img)
                frame_images_payload.append(make_frame_image(url, "last_frame"))

            # Build input_references list
            input_refs_payload: list[dict] = []
            for ri in ref_images_pil:
                url = _pil_to_data_url(ri)
                input_refs_payload.append(make_input_reference(url))

            actual_seed = seed if not use_random_seed else random.randint(1, 1_000_000)

            # Auto-detect aspect ratio from first frame image if enabled
            effective_aspect_ratio = aspect_ratio
            if use_auto_aspect:
                ref_img_for_ar = first_frame_img or last_frame_img or (ref_images_pil[0] if ref_images_pil else None)
                if ref_img_for_ar is not None:
                    effective_aspect_ratio = _get_image_aspect_ratio(ref_img_for_ar)

            req = VideoRequest(
                model=model_id,
                prompt=prompt.strip(),
                duration=duration,
                resolution=resolution,
                aspect_ratio=effective_aspect_ratio,
                generate_audio=generate_audio,
                seed=actual_seed,
                frame_images=frame_images_payload,
                input_references=input_refs_payload,
            )

            # Reset state
            st.session_state.video_gen_result = None
            st.session_state.video_gen_saved_path = None
            st.session_state.video_gen_error = None
            st.session_state.video_gen_status = "Starting..."
            st.session_state.video_gen_running = True
            st.session_state.video_gen_start_time = time.time()
            st.session_state.video_gen_log = []
            st.session_state.video_gen_prompt_used = prompt.strip()
            st.session_state.video_gen_model_used = model_id
            st.session_state.video_gen_params_used = {
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "generate_audio": generate_audio,
                "seed": actual_seed,
            }

            # Submit — fast, just sends the POST request
            _submit_job(api_key, req)
            st.rerun()

        # ── Manual check button ──────────────────────────────────────────────
        job_id = st.session_state.get("video_gen_job_id")
        job_status = st.session_state.get("video_gen_status")
        if job_id and job_status not in (None, "completed", "error"):
            submitted_at = st.session_state.get("video_gen_start_time")
            elapsed = int(time.time() - submitted_at) if submitted_at else 0
            st.info(f"⏳ Job `{job_id}` — last status: **{job_status}** ({elapsed}s ago)")
            st.caption("The job is running on OpenRouter servers. Press the button below when you want to check if it's ready — no need to wait here.")
            if st.button("🔄 Check job status", key="video_check_btn", use_container_width=True, type="primary"):
                with st.spinner("Polling..."):
                    _do_poll(api_key)
                st.rerun()

        # ── Error display ────────────────────────────────────────────────────
        if st.session_state.video_gen_error:
            st.error(f"❌ Generation failed: {st.session_state.video_gen_error}")

        # ── Debug log panel ──────────────────────────────────────────────────
        gen_log: list = st.session_state.get("video_gen_log") or []
        if gen_log or st.session_state.video_gen_status:
            with st.expander("🔍 API Debug Log", expanded=bool(st.session_state.video_gen_error)):
                st.caption("Full request/response trace for every API call in the current generation.")
                if st.session_state.video_gen_status:
                    st.markdown(f"**Current status:** `{st.session_state.video_gen_status}`")
                for entry in gen_log:
                    ts = entry.get("_ts", "")
                    event = entry.get("event", "?")
                    data = entry.get("data")
                    label = f"`{ts}` — **{event}**"
                    if event == "submit_payload":
                        with st.expander(label, expanded=False):
                            st.json(data)
                    elif event == "submit_response":
                        color = "green" if not data.get("error") else "red"
                        with st.expander(label, expanded=True):
                            st.json(data)
                    elif event.startswith("poll_") and not event.endswith("error"):
                        with st.expander(label, expanded=False):
                            st.json(data)
                    else:
                        # errors, download events, etc — always expanded
                        with st.expander(label, expanded=True):
                            if isinstance(data, (dict, list)):
                                st.json(data)
                            else:
                                st.code(str(data))

        # ── Video result display ─────────────────────────────────────────────
        video_bytes: bytes | None = st.session_state.video_gen_result

        if video_bytes:
            st.success("✅ Video generated successfully!")
            st.video(video_bytes)

            # Generation details
            params = st.session_state.video_gen_params_used
            with st.expander("ℹ️ Generation Details", expanded=True):
                st.markdown(f"**Model:** `{st.session_state.video_gen_model_used}`")
                st.markdown(f"**Prompt:** {st.session_state.video_gen_prompt_used[:300]}")
                for k, v in params.items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                cost = estimate_cost(
                    st.session_state.video_gen_model_used,
                    params.get("duration", 5),
                    params.get("resolution", "720p"),
                    params.get("generate_audio", True),
                )
                if cost:
                    st.markdown(f"**Estimated Cost:** {cost}")

            # Auto-save already happened once in _do_poll — just show where it went.
            saved_path = st.session_state.get("video_gen_saved_path")
            if saved_path:
                st.caption(f"💾 Auto-saved to `{saved_path}`")
            elif auto_save_video:
                # Fallback: save now if it wasn't saved during polling (e.g. old job)
                try:
                    saved_path = _save_video(
                        video_bytes,
                        st.session_state.video_gen_prompt_used,
                        st.session_state.video_gen_model_used,
                    )
                    st.session_state.video_gen_saved_path = saved_path
                    st.caption(f"💾 Auto-saved to `{saved_path}`")
                except Exception as e:
                    st.warning(f"Auto-save failed: {e}")

            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Video (MP4)",
                data=video_bytes,
                file_name=f"video_{timestamp}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            # Add to history
            history_entry = {
                "video_bytes": video_bytes,
                "prompt": st.session_state.video_gen_prompt_used,
                "model": st.session_state.video_gen_model_used,
                "params": params.copy(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "saved_path": saved_path,
            }
            # Avoid duplicates: only add if not already the latest
            hist = st.session_state.video_generated_history
            if not hist or hist[-1].get("timestamp") != history_entry["timestamp"]:
                st.session_state.video_generated_history.append(history_entry)
                if len(st.session_state.video_generated_history) > 10:
                    st.session_state.video_generated_history.pop(0)

        # ── Quick prompt generator (video) ──────────────────────────────────
        st.divider()
        with st.expander("⚡ Quick Video Prompt Generator (LLM)", expanded=False):
            st.caption("Describe what you want in plain language — the LLM will craft a proper video prompt.")
            draft = st.text_area(
                "Draft description",
                height=80,
                key="video_qpg_draft",
                placeholder="A woman walking in a rainy Tokyo street at night, neon reflections...",
            )
            qpg_model = st.selectbox(
                "LLM Model",
                options=["x-ai/grok-4.3", "x-ai/grok-4.2", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4.5"],
                key="video_qpg_model",
            )

            # x-ai/grok-4.3
            if st.button("Generate Prompt", key="video_qpg_btn") and draft.strip():
                system = (
                    "You are an expert video-prompt engineer. Given a brief description, "
                    "write a single, rich, cinematic text-to-video prompt (2-4 sentences) "
                    "suitable for state-of-the-art video generation models. Include: scene, "
                    "subject action, camera movement, lighting, mood, and visual style. "
                    "Output only the prompt, no preamble, no quotes."
                )
                try:
                    with st.spinner("Generating prompt..."):
                        result = llm_inference(draft.strip(), system=system, model=qpg_model, temperature=0.85)
                    if result:
                        st.session_state["video_qpg_result"] = result.strip()
                except Exception as e:
                    st.error(f"❌ {e}")

            if st.session_state.get("video_qpg_result"):
                qpg_res = st.text_area(
                    "Generated Prompt (copy to use above)",
                    value=st.session_state["video_qpg_result"],
                    height=120,
                    key="video_qpg_result_area",
                )
                if st.button("📋 Use this prompt", key="video_qpg_use_btn"):
                    st.session_state["video_llm_enhanced_prompt"] = qpg_res
                    st.rerun()

    # ── Extract last frame from an uploaded video ───────────────────────────
    st.divider()
    _last_frame_extractor_box()

    # ── Merge / concatenate multiple videos ─────────────────────────────────
    st.divider()
    _video_merger_box()

    # ── Generation History ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📼 Video Generation History")
    hist = st.session_state.video_generated_history
    if not hist:
        st.caption("No videos generated yet in this session.")
    else:
        for i, entry in enumerate(reversed(hist)):
            idx = len(hist) - i
            label = f"#{idx} — {entry['timestamp']} — {entry['model'].split('/')[-1]} — {entry['prompt'][:60]}..."
            with st.expander(label, expanded=(i == 0)):
                st.video(entry["video_bytes"])
                st.markdown(f"**Model:** `{entry['model']}`")
                st.markdown(f"**Prompt:** {entry['prompt']}")
                for k, v in entry["params"].items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                if entry.get("saved_path"):
                    st.markdown(f"**Saved to:** `{entry['saved_path']}`")
                cost = estimate_cost(
                    entry["model"],
                    entry["params"].get("duration", 5),
                    entry["params"].get("resolution", "720p"),
                    entry["params"].get("generate_audio", True),
                )
                if cost:
                    st.markdown(f"**Estimated Cost:** {cost}")
                ts2 = entry["timestamp"].replace(":", "-").replace(" ", "_")
                st.download_button(
                    f"⬇️ Download #{idx}",
                    data=entry["video_bytes"],
                    file_name=f"video_{ts2}.mp4",
                    mime="video/mp4",
                    key=f"video_hist_dl_{i}",
                )

    # ── Recover past jobs ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Recover Past Jobs")
    st.caption(
        f"Jobs are logged locally in `{JOBS_LOG_FILE}`. "
        "OpenRouter does not expose a list-jobs API — every job submitted from this app is saved here automatically. "
        "Poll any pending job to check if it completed and download the result."
    )

    if not api_key:
        st.warning("⚠️ API key required to poll/download jobs.")
    else:
        auto_fetch = st.session_state.pop("video_scroll_to_recover", False)

        col_r1, col_r2 = st.columns([1, 3])
        with col_r1:
            load_btn = st.button("📋 Load from log file", key="video_recover_fetch_btn", use_container_width=True)
        with col_r2:
            manual_polling_url = st.text_input(
                "Or enter polling URL / job ID manually",
                key="video_recover_manual_id",
                placeholder="https://openrouter.ai/api/v1/videos/gen_01j…  or just  gen_01j…",
            )
            manual_fetch_btn = st.button("🔍 Add & poll", key="video_recover_manual_btn")

        if load_btn or auto_fetch:
            persisted = _load_persisted_jobs()
            if persisted:
                st.session_state["video_recover_jobs"] = persisted
                st.success(f"Loaded {len(persisted)} job(s) from `{JOBS_LOG_FILE}`.")
            else:
                st.info(f"No jobs found in `{JOBS_LOG_FILE}` yet. Jobs are saved automatically when you generate.")

        # Add a job manually by polling URL or bare job ID
        if manual_fetch_btn and manual_polling_url.strip():
            raw = manual_polling_url.strip()
            poll_url = raw if raw.startswith("http") else f"https://openrouter.ai/api/v1/videos/{raw}"
            with st.spinner(f"Polling {poll_url}..."):
                try:
                    r = requests.get(
                        poll_url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=30,
                    )
                    r.raise_for_status()
                    polled = r.json()
                    # Normalise: polling response may have different shape than persisted entry
                    jid = polled.get("id") or polled.get("job_id") or raw
                    entry = {
                        "job_id": jid,
                        "polling_url": poll_url,
                        "model": polled.get("model", "?"),
                        "prompt": polled.get("prompt", ""),
                        "status": polled.get("status", "?"),
                        "submitted_at": polled.get("created_at", ""),
                        "_polled_response": polled,
                    }
                    existing: list = st.session_state.get("video_recover_jobs", [])
                    ids = [j.get("job_id") for j in existing]
                    if jid in ids:
                        existing[ids.index(jid)] = entry
                    else:
                        existing = [entry] + existing
                    st.session_state["video_recover_jobs"] = existing
                except Exception as e:
                    st.error(f"❌ Poll failed: {e}")

        # ── Job list display ─────────────────────────────────────────────────
        jobs: list = st.session_state.get("video_recover_jobs", [])
        if not jobs:
            st.caption("No jobs loaded yet. Click **Load from log file** or add a polling URL manually.")
        else:
            st.caption(f"**{len(jobs)}** job(s) — newest first.")
            STATUS_ICON = {"completed": "✅", "failed": "❌", "cancelled": "🚫",
                           "submitted": "🕐", "processing": "⏳", "queued": "🕐", "error": "❌"}

            for j_idx, job in enumerate(reversed(jobs)):
                jid     = job.get("job_id", "?")
                jstatus = job.get("status", "?")
                jmodel  = (job.get("model") or "?").split("/")[-1]
                jprompt = (job.get("prompt") or "")[:70]
                jsub    = job.get("submitted_at", "")[:16]
                icon    = STATUS_ICON.get(jstatus, "❓")
                label   = f"{icon} `{jid}` — {jstatus} — {jmodel} — {jprompt or '(no prompt)'} {jsub}"

                with st.expander(label, expanded=False):
                    # Show stored metadata (exclude bulky polled response by default)
                    meta = {k: v for k, v in job.items() if k != "_polled_response"}
                    st.json(meta)
                    if job.get("saved_path") and os.path.exists(job["saved_path"]):
                        st.success(f"Already saved locally: `{job['saved_path']}`")
                        with open(job["saved_path"], "rb") as fv:
                            st.video(fv.read())

                    poll_url = job.get("polling_url", "")

                    # Poll / re-check button (always available for non-completed jobs)
                    if jstatus not in ("completed",) and poll_url:
                        if st.button(f"🔄 Poll status now", key=f"vr_poll_{j_idx}"):
                            with st.spinner("Polling..."):
                                try:
                                    r = requests.get(
                                        poll_url,
                                        headers={"Authorization": f"Bearer {api_key}"},
                                        timeout=30,
                                    )
                                    r.raise_for_status()
                                    polled = r.json()
                                    new_status = polled.get("status", jstatus)
                                    job["status"] = new_status
                                    job["_polled_response"] = polled
                                    _update_persisted_job(jid, new_status)
                                    # grab unsigned_urls if present
                                    if polled.get("unsigned_urls"):
                                        job["unsigned_urls"] = polled["unsigned_urls"]
                                    real_idx = len(jobs) - 1 - j_idx
                                    jobs[real_idx] = job
                                    st.session_state["video_recover_jobs"] = jobs
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Poll failed: {e}")

                    # Download if URLs available
                    urls = job.get("unsigned_urls") or job.get("_polled_response", {}).get("unsigned_urls") or []
                    if urls:
                        for u_idx, video_url in enumerate(urls):
                            if video_url.startswith("/"):
                                video_url = f"https://openrouter.ai{video_url}"
                            if st.button(f"⬇️ Download & save video", key=f"vr_dl_{j_idx}_{u_idx}"):
                                with st.spinner("Downloading..."):
                                    try:
                                        r = requests.get(
                                            video_url,
                                            headers={"Authorization": f"Bearer {api_key}"},
                                            stream=True, timeout=120,
                                        )
                                        r.raise_for_status()
                                        vbytes = r.content
                                        os.makedirs("outputs/videos", exist_ok=True)
                                        fname = f"recovered_{jid}_{u_idx}.mp4"
                                        fpath = os.path.join("outputs/videos", fname)
                                        with open(fpath, "wb") as fv:
                                            fv.write(vbytes)
                                        job["status"] = "completed"
                                        job["saved_path"] = fpath
                                        real_idx = len(jobs) - 1 - j_idx
                                        jobs[real_idx] = job
                                        st.session_state["video_recover_jobs"] = jobs
                                        _update_persisted_job(jid, "completed", fpath)
                                        st.success(f"Saved to `{fpath}`")
                                        st.video(vbytes)
                                        st.download_button(
                                            "⬇️ Save to computer",
                                            data=vbytes,
                                            file_name=fname,
                                            mime="video/mp4",
                                            key=f"vr_save_{j_idx}_{u_idx}",
                                        )
                                    except Exception as e:
                                        st.error(f"❌ Download failed: {e}")
