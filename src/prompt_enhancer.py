"""Shared LLM prompt enhance/fix widget for Streamlit pages."""
import streamlit as st
from src.llm_lite import llm_inference

LLM_ENHANCE_MODEL = "x-ai/grok-4.3"

# LLM_ENHANCE_SYSTEM = (
#     "You are an expert image-prompt engineer. Expand and enrich the user's prompt "
#     "for a text-to-image model: add vivid visual detail, style, lighting, composition, "
#     "and camera cues while preserving the original subject and intent. "
#     "Return ONLY the enhanced prompt as plain text, no preamble, no quotes, no explanation."
# )

LLM_ENHANCE_SYSTEM = (
    "You are an expert image-prompt engineer. Lightly enhance the user's prompt by "
    "adding modest visual detail (style, lighting, composition) ONLY where the "
    "prompt is sparse. Preserve the original subject, intent, structure, and length. "
    "If the prompt is an editing instruction (contains verbs like change, replace, "
    "remove, add) or is already detailed, return it nearly unchanged. "
    "Do NOT invent new subjects or scene elements. Keep the original language. "
    "Return ONLY the enhanced prompt as plain text, no preamble, no quotes, no explanation."
)

LLM_FIX_SYSTEM = (
    "You are a proofreader. Fix ONLY grammar, spelling, punctuation, and syntax "
    "errors in the user's image prompt. Do NOT add new content. Do NOT change style, "
    "tone, vocabulary, or meaning. Do NOT translate: keep the EXACT same language as "
    "the input — if the input is Italian, output Italian; if Spanish, output Spanish; "
    "if mixed, keep it mixed. Preserve technical tags, model tokens, LoRA triggers, "
    "and artist names verbatim. If there are no errors, return the prompt unchanged. "
    "Return ONLY the corrected prompt as plain text, no preamble, no quotes, no explanation."
)

LLM_RESTYLE_SYSTEM = (
    "You are an expert image-prompt editor performing a CONSERVATIVE RESTYLE. "
    "You will receive an input formatted as:\n"
    "ORIGINAL PROMPT:\n<the prompt to restyle>\n\n"
    "USER INSTRUCTIONS:\n<short restyle directions, possibly in any language>\n\n"
    "Your task: rewrite the ORIGINAL PROMPT applying ONLY the changes implied by the "
    "USER INSTRUCTIONS. The user's instructions may be written in ANY language "
    "(Italian, English, Spanish, French, German, etc.) — understand them regardless "
    "of language, but ALWAYS write the output prompt in English (the language used "
    "by text-to-image models).\n\n"
    "Strict rules:\n"
    "- Be CONSERVATIVE: preserve subject, composition, framing, characters, setting, "
    "and overall intent of the original prompt.\n"
    "- Modify ONLY the aspects the user explicitly asks to change (e.g. style, "
    "lighting, palette, mood, medium, era, clothing detail). Leave everything else "
    "untouched and intact.\n"
    "- Do NOT add new subjects, objects, characters, or scene elements that the user "
    "did not request.\n"
    "- Do NOT remove existing elements unless the user explicitly asks to remove them.\n"
    "- Do NOT expand or enrich the prompt with extra creative detail beyond what the "
    "user requested — this is a restyle, not an enhancement.\n"
    "- Keep the original prompt's structure and length as close as possible to the "
    "input; only swap/adjust the parts affected by the user instructions.\n"
    "- If the user instructions are empty or unclear, return the ORIGINAL PROMPT "
    "unchanged.\n\n"
    "Return ONLY the restyled prompt as plain text — no preamble, no quotes, no "
    "explanation, no language tags."
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

    def _run_llm(system_prompt: str, label: str, user_payload: str | None = None):
        src = (prompt or "").strip()
        if not src:
            st.warning("⚠️ Prompt is empty.")
            return
        try:
            with st.spinner(f"{label} via {LLM_ENHANCE_MODEL}..."):
                result = llm_inference(
                    prompt=user_payload if user_payload is not None else src,
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

    restyle_key = f"{session_key}_restyle_instructions"
    restyle_instructions = st.text_input(
        "🎨 Restyle instructions (any language)",
        key=restyle_key,
        placeholder="e.g. make it watercolor / stile anni '80 / iluminación nocturna",
        help="Short directions for the Restyle button. The original prompt is kept "
             "intact except for the aspects you mention here.",
    )

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button("✨ Enhance", key=f"{session_key}_btn_enhance",
                     help=f"Expand the prompt via {LLM_ENHANCE_MODEL}"):
            _run_llm(LLM_ENHANCE_SYSTEM, "Enhancing")
    with cols[1]:
        if st.button("🛠️ Fix", key=f"{session_key}_btn_fix",
                     help="Fix grammar/syntax errors only"):
            _run_llm(LLM_FIX_SYSTEM, "Fixing")
    with cols[2]:
        if st.button("🎨 Restyle", key=f"{session_key}_btn_restyle",
                     help="Conservative restyle using only your instructions above"):
            instr = (restyle_instructions or "").strip()
            if not instr:
                st.warning("⚠️ Enter restyle instructions first.")
            else:
                payload = (
                    f"ORIGINAL PROMPT:\n{(prompt or '').strip()}\n\n"
                    f"USER INSTRUCTIONS:\n{instr}"
                )
                _run_llm(LLM_RESTYLE_SYSTEM, "Restyling", user_payload=payload)
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
