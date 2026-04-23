"""Shared LLM prompt enhance/fix widget for Streamlit pages."""
import streamlit as st
from src.llm_lite import llm_inference

LLM_ENHANCE_MODEL = "x-ai/grok-4"

LLM_ENHANCE_SYSTEM = (
    "You are an expert image-prompt engineer. Expand and enrich the user's prompt "
    "for a text-to-image model: add vivid visual detail, style, lighting, composition, "
    "and camera cues while preserving the original subject and intent. "
    "Return ONLY the enhanced prompt as plain text, no preamble, no quotes, no explanation."
)

LLM_FIX_SYSTEM = (
    "You are a proofreader. Fix grammar, spelling, and syntax errors in the user's "
    "image prompt. Do NOT add new content, do NOT change style, tone, or meaning — "
    "only correct mistakes. Preserve the original language. "
    "Return ONLY the corrected prompt as plain text, no preamble, no quotes, no explanation."
)


def render_prompt_enhancer(prompt: str, session_key: str = "llm_enhanced_prompt") -> str:
    """
    Render Enhance / Fix / Clear buttons and (if active) an editable text area
    with the LLM-processed prompt. The editable text overrides `prompt`.

    Args:
        prompt: current prompt from the upstream text_area
        session_key: unique session_state key (use different keys on different pages)

    Returns:
        The prompt to actually use for generation.
    """

    def _run_llm(system_prompt: str, label: str):
        src = (prompt or "").strip()
        if not src:
            st.warning("⚠️ Prompt is empty.")
            return
        try:
            with st.spinner(f"{label} via {LLM_ENHANCE_MODEL}..."):
                result = llm_inference(
                    prompt=src,
                    system=system_prompt,
                    model=LLM_ENHANCE_MODEL,
                    temperature=0.7,
                )
            if result and result.strip():
                st.session_state[session_key] = result.strip()
                st.rerun()
            else:
                st.warning("⚠️ LLM returned empty response.")
        except Exception as e:
            st.error(f"❌ LLM {label.lower()} failed: {e}")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("✨ Enhance", key=f"{session_key}_btn_enhance",
                     help=f"Expand the prompt via {LLM_ENHANCE_MODEL}"):
            _run_llm(LLM_ENHANCE_SYSTEM, "Enhancing")
    with cols[1]:
        if st.button("🛠️ Fix", key=f"{session_key}_btn_fix",
                     help="Fix grammar/syntax errors only"):
            _run_llm(LLM_FIX_SYSTEM, "Fixing")
    with cols[2]:
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
