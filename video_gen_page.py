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
import threading
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


# ---------------------------------------------------------------------------
# BACKGROUND GENERATION THREAD
# ---------------------------------------------------------------------------
def _run_generation_thread(
    api_key: str,
    req: VideoRequest,
    result_key: str,
    status_key: str,
    error_key: str,
):
    """Runs in background thread. Writes result to st.session_state when done."""
    try:
        client = OpenRouterVideoClient(api_key=api_key)

        def _update_status(msg: str):
            st.session_state[status_key] = msg

        _update_status("Submitting job...")
        sub = client.submit(req)
        job_id = sub.get("id", "?")
        polling_url = sub.get("polling_url", "")
        _update_status(f"Job {job_id} submitted — polling...")

        start = time.time()
        terminal = {"completed", "failed", "cancelled", "expired"}
        while True:
            status_resp = client.poll(polling_url)
            s = status_resp.get("status")
            elapsed = int(time.time() - start)
            _update_status(f"Status: {s} ({elapsed}s elapsed)")
            if s in terminal:
                if s == "completed":
                    urls = status_resp.get("unsigned_urls") or []
                    if urls:
                        video_url = urls[0]
                        if video_url.startswith("/"):
                            video_url = f"https://openrouter.ai{video_url}"
                        resp = requests.get(video_url, stream=True, timeout=120)
                        resp.raise_for_status()
                        video_bytes = resp.content
                        st.session_state[result_key] = video_bytes
                        _update_status("completed")
                    else:
                        st.session_state[error_key] = "Completed but no video URL returned."
                        _update_status("error")
                else:
                    err = status_resp.get("error") or s
                    st.session_state[error_key] = f"Job {s}: {err}"
                    _update_status("error")
                break
            if time.time() - start > 900:
                st.session_state[error_key] = "Timeout: job did not complete within 15 minutes."
                _update_status("error")
                break
            time.sleep(8)
    except Exception as e:
        st.session_state[error_key] = str(e)
        st.session_state[status_key] = "error"


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------
def show_video_generator_page():
    st.title("🎬 OpenRouter Video Generator")
    st.markdown("Generate videos using OpenRouter video models with full parameter control.")

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

        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            options=VIDEO_ASPECT_RATIOS,
            index=0,  # 16:9 default
            key="video_aspect_ratio",
        )

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

            req = VideoRequest(
                model=model_id,
                prompt=prompt.strip(),
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                generate_audio=generate_audio,
                seed=actual_seed,
                frame_images=frame_images_payload,
                input_references=input_refs_payload,
            )

            # Reset state
            st.session_state.video_gen_result = None
            st.session_state.video_gen_error = None
            st.session_state.video_gen_status = "Starting..."
            st.session_state.video_gen_running = True
            st.session_state.video_gen_prompt_used = prompt.strip()
            st.session_state.video_gen_model_used = model_id
            st.session_state.video_gen_params_used = {
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "generate_audio": generate_audio,
                "seed": actual_seed,
            }

            thread = threading.Thread(
                target=_run_generation_thread,
                args=(
                    api_key,
                    req,
                    "video_gen_result",
                    "video_gen_status",
                    "video_gen_error",
                ),
                daemon=True,
            )
            thread.start()
            st.rerun()

        # ── Status / progress ────────────────────────────────────────────────
        status = st.session_state.video_gen_status
        running = st.session_state.video_gen_running

        if running and status not in (None, "completed", "error"):
            st.info(f"⏳ {status}")
            st.caption("Video generation can take 1–5 minutes. This page auto-refreshes every 8s.")
            time.sleep(8)
            # Check if thread finished
            if st.session_state.video_gen_status in ("completed", "error"):
                st.session_state.video_gen_running = False
            st.rerun()

        if status == "completed" or (status and "completed" in str(status)):
            st.session_state.video_gen_running = False

        if status == "error" or st.session_state.video_gen_error:
            st.session_state.video_gen_running = False

        # ── Error display ────────────────────────────────────────────────────
        if st.session_state.video_gen_error:
            st.error(f"❌ Generation failed: {st.session_state.video_gen_error}")

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

            # Auto-save
            saved_path = None
            if auto_save_video:
                try:
                    saved_path = _save_video(
                        video_bytes,
                        st.session_state.video_gen_prompt_used,
                        st.session_state.video_gen_model_used,
                    )
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

        elif not running and status is not None and status not in ("completed", "error") and "completed" not in str(status):
            pass  # nothing to show yet

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
