from pyperclip import copy
import streamlit as st
from datetime import datetime
import random
from io import BytesIO
import base64
import os
from PIL import Image, ImageOps, PngImagePlugin
import json
import yaml

from google import genai
from google.genai import types

# Page configuration (only used when running standalone)
if __name__ == "__main__" or "google_page" not in str(st.session_state.get("_page_loaded", "")):
    try:
        st.set_page_config(
            page_title="Google AI Image Generator",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass

# Model definitions - Google AI Studio image-capable models
"""

"""
GOOGLE_IMAGE_MODELS = {
    "gemini-2.5-flash-image":"gemini-2.5-flash-image",
    "gemini-3-pro-image-preview":"gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview":"gemini-3.1-flash-image-preview",
    "imagen-4.0-generate-001":"imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001":"imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001":"imagen-4.0-fast-generate-001",
}
default_model = "gemini-2.5-flash-image"

# Models that support img2img (reference images)
MODELS_WITH_IMG2IMG =  GOOGLE_IMAGE_MODELS.keys()

# Imagen models use a different API
IMAGEN_MODELS = {
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
}

ASPECT_RATIOS = {
    "1:1 (1024x1024)": "1:1",
    "2:3 (832x1248)": "2:3",
    "3:2 (1248x832)": "3:2",
    "3:4 (864x1184)": "3:4",
    "4:3 (1184x864)": "4:3",
    "4:5 (896x1152)": "4:5",
    "5:4 (1152x896)": "5:4",
    "9:16 (768x1344)": "9:16",
    "16:9 (1344x768)": "16:9",
    "21:9 (1536x672)": "21:9"
}

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
]

# Helper functions
def load_google_api_key():
    """Load Google AI API key from api_keys.json or session state"""
    if 'google_api_key' in st.session_state and st.session_state.google_api_key:
        return st.session_state.google_api_key

    if os.path.exists("api_keys.json"):
        try:
            with open("api_keys.json", 'r') as f:
                api_dict = json.load(f)
                return api_dict.get("googleai", api_dict.get("gemini", ""))
                # return api_dict.get("googleai", "")
        except:
            return ""
    return ""

def load_prompts_from_yaml(file_path="prompts.yaml"):
    root_path = "prompts/"
    file_path = os.path.join(root_path, file_path)
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        return {}

def load_prompts_from_json(file_path="prompts.json"):
    root_path = "prompts/"
    file_path = os.path.join(root_path, file_path)
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading prompts from JSON: {str(e)}")
        return {}

def flatten_json_prompts(prompts_data):
    """Flatten JSON prompts into section/category/prompt entries."""
    flattened = []
    if not isinstance(prompts_data, dict):
        return flattened

    def dump_prompt(value):
        return json.dumps(value, indent=2, ensure_ascii=False)

    processed_sections = set()

    def process_section(section_key, section_label):
        section_data = prompts_data.get(section_key)
        if not isinstance(section_data, dict):
            return
        processed_sections.add(section_key)
        for category_key, category_value in section_data.items():
            if isinstance(category_value, dict) and category_value:
                for prompt_name, prompt_value in category_value.items():
                    flattened.append({
                        'section': section_label,
                        'category': category_key,
                        'prompt_name': prompt_name,
                        'prompt': dump_prompt(prompt_value),
                        'index': 0
                    })
            else:
                flattened.append({
                    'section': section_label,
                    'category': category_key,
                    'prompt_name': category_key,
                    'prompt': dump_prompt(category_value),
                    'index': 0
                })

    process_section("create_prompts", "Create")
    process_section("edit_prompts", "Edit")

    for key, value in prompts_data.items():
        if key not in processed_sections:
            if isinstance(value, dict):
                process_section(key, key)
            else:
                flattened.append({
                    'section': 'Other',
                    'category': key,
                    'prompt_name': key,
                    'prompt': dump_prompt(value),
                    'index': 0
                })
    return flattened

def flatten_prompts(prompts_data):
    """Flatten YAML prompts into a list of entries."""
    flattened = []
    if not isinstance(prompts_data, dict):
        return flattened
    for section_key, section_data in prompts_data.items():
        if isinstance(section_data, dict):
            for category_key, category_data in section_data.items():
                if isinstance(category_data, list):
                    for idx, prompt in enumerate(category_data):
                        if isinstance(prompt, str):
                            flattened.append({
                                'section': section_key,
                                'category': category_key,
                                'prompt': prompt,
                                'index': idx
                            })
                elif isinstance(category_data, dict):
                    for sub_key, sub_data in category_data.items():
                        if isinstance(sub_data, list):
                            for idx, prompt in enumerate(sub_data):
                                if isinstance(prompt, str):
                                    flattened.append({
                                        'section': section_key,
                                        'category': f"{category_key}/{sub_key}",
                                        'prompt': prompt,
                                        'index': idx
                                    })
    return flattened

def get_google_client(api_key):
    """Create Google GenAI client"""
    return genai.Client(api_key=api_key)

def encode_image_to_base64(image):
    """Encode PIL Image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_aspect_ratio(image):
    """Calculate closest aspect ratio from image dimensions"""
    width, height = image.size
    ratio = width / height
    aspect_ratios = {
        "1:1": 1.0, "2:3": 2/3, "3:2": 3/2, "3:4": 3/4,
        "4:3": 4/3, "4:5": 4/5, "5:4": 5/4, "9:16": 9/16,
        "16:9": 16/9, "21:9": 21/9
    }
    closest_ratio = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
    return closest_ratio[0]

def resize_image(image, max_size=1024):
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def pil_to_bytes(image, format="PNG"):
    """Convert PIL Image to bytes"""
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def generate_image_google(prompt, api_key, model_key, aspect_ratio, seed,
                          reference_images=None, use_image_aspect_ratio=False,
                          max_image_size=1024):
    """Generate image using Google AI Studio API"""

    if not seed:
        seed = random.randint(1, 1000000)

    client = get_google_client(api_key)
    model_id = GOOGLE_IMAGE_MODELS[model_key]

    # Auto-detect aspect ratio from reference image
    if use_image_aspect_ratio and reference_images:
        aspect_ratio = get_image_aspect_ratio(reference_images[0])

    # Check if this is an Imagen model (different API)
    if model_key in IMAGEN_MODELS:
        return _generate_imagen(client, model_id, prompt, aspect_ratio, seed)

    # Gemini models - use generate_content with image modality
    contents = []

    # Add reference images if provided and model supports it
    if reference_images and model_key in MODELS_WITH_IMG2IMG:
        for ref_img in reference_images:
            resized = resize_image(ref_img, max_image_size)
            img_bytes = pil_to_bytes(resized, format="JPEG")
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

    # Add the text prompt
    contents.append(prompt)

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=SAFETY_SETTINGS,
            seed=seed,
        )
    )

    # Extract images from response
    if not response.candidates or not response.candidates[0].content.parts:
        return None, aspect_ratio, None

    images = [
        part for part in response.candidates[0].content.parts
        if part.inline_data is not None
    ]

    # Extract any text response
    text_parts = [
        part.text for part in response.candidates[0].content.parts
        if hasattr(part, 'text') and part.text
    ]
    response_text = "\n".join(text_parts) if text_parts else None

    if not images:
        return None, aspect_ratio, response_text

    # Convert first image to PIL
    image_data = images[0].inline_data.data
    image = Image.open(BytesIO(image_data))

    return image, aspect_ratio, response_text


def _generate_imagen(client, model_id, prompt, aspect_ratio, seed):
    """Generate image using Imagen models (different API)"""
    try:
        response = client.models.generate_images(
            model=model_id,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                seed=seed,
                safety_filter_level="BLOCK_NONE",
            )
        )

        if response.generated_images:
            img_data = response.generated_images[0].image.image_bytes
            image = Image.open(BytesIO(img_data))
            return image, aspect_ratio, None

        return None, aspect_ratio, None
    except Exception as e:
        # Fallback: try generate_content for Imagen models too
        raise e


def save_image_with_metadata(image, prompt, model_name, seed, aspect_ratio,
                             output_folder="outputs", reduce_quality=False):
    """Save image with metadata to outputs folder"""
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Model", model_name)
    metadata.add_text("Seed", str(seed))
    metadata.add_text("Aspect_Ratio", aspect_ratio)
    metadata.add_text("Provider", "Google AI Studio")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_short = prompt[:30].replace(" ", "_").replace("\n", "_")
    prompt_short = prompt_short.replace("__", "_")
    prompt_short = "".join(c for c in prompt_short if c.isalnum() or c in ('_', '-'))

    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name

    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, f"{prompt_short}_{model_name_short}_{timestamp}.png")
    image.save(filename, pnginfo=metadata)
    return filename

def create_comparison_image(generated_image, reference_images, max_refs=3, prompt="", model="", seed=None):
    """Create a side-by-side comparison image with reference and generated images"""
    if not reference_images:
        return None

    from PIL import ImageDraw, ImageFont

    ref_images = reference_images[:max_refs]
    gen_width, gen_height = generated_image.size

    resized_refs = []
    total_ref_width = 0
    for ref_img in ref_images:
        aspect = ref_img.width / ref_img.height
        new_height = gen_height
        new_width = int(new_height * aspect)
        resized_ref = ref_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_refs.append(resized_ref)
        total_ref_width += new_width

    padding = 20
    text_area_height = 100
    total_width = total_ref_width + gen_width + padding * (len(resized_refs) + 1)

    comparison = Image.new('RGB', (total_width, gen_height + padding * 2 + text_area_height), color='white')

    x_offset = padding
    for ref_img in resized_refs:
        comparison.paste(ref_img, (x_offset, padding))
        x_offset += ref_img.width + padding

    comparison.paste(generated_image, (x_offset, padding))

    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    text_y = gen_height + padding * 2 + 10
    text_x = padding

    metadata_text = [
        f"Model: {model} (Google AI Studio)",
        f"Seed: {seed if seed else 'Random'}"
    ]

    for i, line in enumerate(metadata_text):
        draw.text((text_x, text_y + i * 20), line, fill='black', font=font)

    prompt_y = text_y + 45
    prompt_text = f"Prompt: {prompt}"
    if len(prompt_text) > 150:
        prompt_text = prompt_text[:150] + "..."
    draw.text((text_x, prompt_y), prompt_text, fill='black', font=font)

    return comparison


def show_google_generator_page():
    """Main function to show the Google AI Studio Image Generator page"""

    # Initialize session state
    if 'google_generated_images' not in st.session_state:
        st.session_state.google_generated_images = []
    if 'google_prompt_history' not in st.session_state:
        st.session_state.google_prompt_history = []
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = load_google_api_key()
    if 'google_prompts_data' not in st.session_state:
        prompts_data = load_prompts_from_yaml()
        st.session_state.google_prompts_data = prompts_data
        st.session_state.google_flattened_prompts = flatten_prompts(prompts_data)
    if 'google_json_prompts_data' not in st.session_state:
        json_prompts_data = load_prompts_from_json()
        st.session_state.google_json_prompts_data = json_prompts_data
        st.session_state.google_flattened_json_prompts = flatten_json_prompts(json_prompts_data)

    # Main UI
    st.title("🎨 Google AI Studio Image Generator")
    st.markdown("Generate images using Google AI Studio API (Gemini & Imagen)")

    # Sidebar - Configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        api_key_input = st.text_input(
            "Google AI API Key",
            value=st.session_state.google_api_key,
            type="password",
            help="Enter your Google AI Studio API key",
            key="google_api_key_input"
        )
        st.session_state.google_api_key = api_key_input

        st.divider()

        # Model selection
        st.subheader("Model Settings")
        selected_model = st.selectbox(
            "Model",
            options=list(GOOGLE_IMAGE_MODELS.keys()),
            index=list(GOOGLE_IMAGE_MODELS.keys()).index(default_model),
            help="Choose the image generation model",
            key="google_model_select"
        )

        # Show model info
        if selected_model in IMAGEN_MODELS:
            st.caption("Imagen model - text-to-image only, no reference images")
        elif selected_model in MODELS_WITH_IMG2IMG:
            st.caption("Gemini model - supports reference images (img2img)")

        # Aspect ratio
        use_auto_aspect = st.checkbox(
            "Auto-detect aspect ratio from reference image",
            value=True,
            help="Automatically use the aspect ratio of the first reference image",
            key="google_auto_aspect"
        )

        aspect_ratio_display = st.selectbox(
            "Aspect Ratio",
            options=list(ASPECT_RATIOS.keys()),
            index=0,
            disabled=use_auto_aspect,
            help="Target aspect ratio for generated image",
            key="google_aspect_ratio"
        )
        aspect_ratio = ASPECT_RATIOS[aspect_ratio_display]

        PROMPTS_FILES = {"base": "prompts",
                         "custom": "prompts_custom",
                         "assets": "prompts_assets"}

        default_prompts = "custom" if (os.path.exists("prompts/prompts_custom.json") or os.path.exists("prompts/prompts_custom.yaml")) else "base"

        st.subheader("Prompt Source Settings")
        selected_prompts = st.selectbox(
            "Prompts Source",
            options=list(PROMPTS_FILES.keys()),
            index=list(PROMPTS_FILES.keys()).index(default_prompts),
            help="Choose the prompt file to use",
            key="google_prompts_source"
        )

        # Seed
        st.subheader("Generation Parameters")
        use_random_seed = st.checkbox("Use random seed", value=True, key="google_random_seed")
        if use_random_seed:
            seed = None
        else:
            seed = st.number_input(
                "Seed",
                min_value=1,
                max_value=1000000,
                value=12345,
                help="Fixed seed for reproducible results",
                key="google_seed_input"
            )

        # Image processing settings
        st.subheader("Reference Image Settings")
        max_image_size = st.slider(
            "Max reference image size (px)",
            min_value=256,
            max_value=2048,
            value=1024,
            step=128,
            help="Maximum dimension for reference images before encoding",
            key="google_max_img_size"
        )

        # Save settings
        st.divider()
        auto_save = st.checkbox(
            "Auto-save generated images",
            value=True,
            help="Automatically save images to outputs folder",
            key="google_auto_save"
        )

        # Stealth Mode
        st.divider()
        stealth_mode = st.checkbox(
            "🕶️ Stealth Mode",
            value=False,
            help="Hide all image thumbnails for privacy",
            key="google_stealth"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Prompt & Reference Images")

        # Prompt source selector
        prompt_source = st.radio(
            "Prompt Source",
            options=["Custom Prompt", "Load from YAML", "Load from JSON"],
            horizontal=True,
            help="Choose to write your own prompt or load from YAML/JSON files",
            key="google_prompt_source"
        )

        prompt = ""

        if prompt_source == "Load from JSON":
            json_prompts_data = load_prompts_from_json(PROMPTS_FILES[selected_prompts] + ".json")
            st.session_state.google_json_prompts_data = json_prompts_data
            st.session_state.google_flattened_json_prompts = flatten_json_prompts(json_prompts_data)

            if st.session_state.google_flattened_json_prompts:
                sections = {}
                for item in st.session_state.google_flattened_json_prompts:
                    section = item.get('section', 'JSON Prompts') or 'JSON Prompts'
                    sections.setdefault(section, []).append(item)

                col_section, col_category = st.columns([1, 2])

                with col_section:
                    section_options = list(sections.keys())
                    selected_section = st.selectbox(
                        "Section", options=section_options,
                        help="Choose between Create or Edit prompts",
                        key="google_json_section_select"
                    )

                section_prompts = sections[selected_section]
                categories = sorted(list(set([p['category'] for p in section_prompts])))

                with col_category:
                    selected_category = st.selectbox(
                        "Category", options=categories,
                        help="Choose a category of prompts",
                        key="google_json_category_select"
                    )

                filtered_prompts = [p for p in section_prompts if p['category'] == selected_category]

                prompt_options = ["Select a prompt..."] + [
                    (p.get('prompt_name') or (p['prompt'][:80] + "..." if len(p['prompt']) > 80 else p['prompt']))
                    for p in filtered_prompts
                ]

                selected_prompt_idx = st.selectbox(
                    "Select Prompt",
                    options=range(len(prompt_options)),
                    format_func=lambda x: prompt_options[x],
                    help="Choose a specific prompt",
                    key="google_json_prompt_select"
                )

                selection_key = f"{selected_section}_{selected_category}_{selected_prompt_idx}"

                if 'google_last_json_selection' not in st.session_state or st.session_state.google_last_json_selection != selection_key:
                    st.session_state.google_last_json_selection = selection_key
                    if selected_prompt_idx > 0:
                        st.session_state['google_json_prompt_input'] = filtered_prompts[selected_prompt_idx - 1]['prompt']
                    else:
                        st.session_state['google_json_prompt_input'] = ""

                if 'google_json_prompt_input' not in st.session_state:
                    st.session_state['google_json_prompt_input'] = ""

                prompt = st.text_area(
                    "Selected Prompt (editable)" if selected_prompt_idx > 0 else "Image Prompt",
                    key="google_json_prompt_input",
                    height=200,
                    placeholder="Select a prompt from the dropdowns above or type here...",
                    help="You can edit the loaded prompt before generating"
                )
            else:
                st.warning("⚠️ No prompts found in prompts.json")
                prompt = st.text_area(
                    "Image Prompt", height=200,
                    placeholder="Describe the image you want to generate...",
                    key="google_custom_prompt_fallback_json"
                )

        elif prompt_source == "Load from YAML":
            prompts_data = load_prompts_from_yaml(f"{PROMPTS_FILES[selected_prompts]}.yaml")
            st.session_state.google_prompts_data = prompts_data

            if st.session_state.google_prompts_data:
                prompts_data = st.session_state.google_prompts_data

                section_keys = {"Create": "create_prompts", "Edit": "edit_prompts"}
                available_sections = [s for s, k in section_keys.items() if k in prompts_data]

                if available_sections:
                    col_section, col_category = st.columns([1, 1])

                    with col_section:
                        selected_section = st.selectbox(
                            "Section", options=available_sections,
                            help="Choose between Create or Edit prompts",
                            key="google_yaml_section_select"
                        )

                    section_key = section_keys[selected_section]
                    section_data = prompts_data.get(section_key, {})
                    categories = list(section_data.keys())

                    with col_category:
                        selected_category = st.selectbox(
                            "Category",
                            options=categories if categories else ["No categories"],
                            help="Choose a category of prompts",
                            key="google_yaml_category_select"
                        )

                    category_data = section_data.get(selected_category, {})

                    if isinstance(category_data, dict):
                        sublevels = list(category_data.keys())
                        has_sublevels = True
                    elif isinstance(category_data, list):
                        sublevels = ["(prompts)"]
                        has_sublevels = False
                    else:
                        sublevels = []
                        has_sublevels = False

                    if has_sublevels and sublevels:
                        selected_sublevel = st.selectbox(
                            "Sublevel", options=sublevels,
                            help="Choose a sublevel of prompts",
                            key="google_yaml_sublevel_select"
                        )
                    else:
                        selected_sublevel = None

                    if has_sublevels and selected_sublevel:
                        sublevel_data = category_data.get(selected_sublevel, [])
                        if isinstance(sublevel_data, dict):
                            prompts_list = []
                            for sub_key, sub_val in sublevel_data.items():
                                if isinstance(sub_val, list):
                                    for p in sub_val:
                                        if isinstance(p, str) and not p.strip().startswith('#'):
                                            prompts_list.append(p.strip())
                                elif isinstance(sub_val, str) and not sub_val.strip().startswith('#'):
                                    prompts_list.append(sub_val.strip())
                        elif isinstance(sublevel_data, list):
                            prompts_list = [p.strip() for p in sublevel_data if isinstance(p, str) and not p.strip().startswith('#')]
                        else:
                            prompts_list = []
                    else:
                        if isinstance(category_data, list):
                            prompts_list = [p.strip() for p in category_data if isinstance(p, str) and not p.strip().startswith('#')]
                        else:
                            prompts_list = []

                    prompt_options = ["Select a prompt..."] + [
                        f"{p[:80]}..." if len(p) > 80 else p for p in prompts_list
                    ]

                    selected_prompt_idx = st.selectbox(
                        "Select Prompt",
                        options=range(len(prompt_options)),
                        format_func=lambda x: prompt_options[x],
                        help="Choose a specific prompt",
                        key="google_yaml_prompt_select"
                    )

                    if selected_prompt_idx > 0:
                        selected_prompt_text = prompts_list[selected_prompt_idx - 1]
                        sublevel_key = selected_sublevel if selected_sublevel else "none"
                        prompt = st.text_area(
                            "Selected Prompt (editable)",
                            value=selected_prompt_text,
                            height=150,
                            key=f"google_yaml_prompt_{selected_section}_{selected_category}_{sublevel_key}_{selected_prompt_idx}",
                            help="You can edit the loaded prompt before generating"
                        )
                    else:
                        prompt = st.text_area(
                            "Image Prompt", height=150,
                            placeholder="Select a prompt from the dropdowns above...",
                            key="google_empty_yaml_prompt"
                        )
                else:
                    st.warning("⚠️ No valid sections found in prompts.yaml")
                    prompt = st.text_area(
                        "Image Prompt", height=150,
                        placeholder="Describe the image you want to generate...",
                        key="google_custom_prompt_fallback_yaml"
                    )
            else:
                st.warning("⚠️ No prompts found in prompts.yaml")
                prompt = st.text_area(
                    "Image Prompt", height=150,
                    placeholder="Describe the image you want to generate...",
                    key="google_custom_prompt_fallback"
                )
        else:
            prompt = st.text_area(
                "Image Prompt", height=150,
                placeholder="Describe the image you want to generate...",
                help="Enter a detailed description of the image you want to create",
                key="google_custom_prompt"
            )

        # Reference images upload
        st.subheader("Reference Images (Optional)")

        if selected_model in IMAGEN_MODELS:
            st.info("Imagen models don't support reference images (text-to-image only)")
            uploaded_files = None
            reference_images = None
        else:
            uploaded_files = st.file_uploader(
                "Upload reference images",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                accept_multiple_files=True,
                help="Upload one or more reference images to guide the generation",
                key="google_ref_upload"
            )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} image(s) uploaded**")
            ref_cols = st.columns(min(len(uploaded_files), 3))
            reference_images = []

            for idx, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                reference_images.append(img)

                with ref_cols[idx % 3]:
                    if not stealth_mode:
                        st.image(img, caption=f"Ref {idx+1}", width=150)
                    st.caption(f"Size: {img.size[0]}x{img.size[1]}")
        elif not uploaded_files:
            reference_images = None

        # Generate button
        generate_btn = st.button(
            "🎨 Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not (prompt and st.session_state.google_api_key),
            key="google_generate_btn"
        )

    with col2:
        st.subheader("🖼️ Generated Image")

        if generate_btn:
            if not st.session_state.google_api_key:
                st.error("❌ Please enter your Google AI API key in the sidebar")
            elif not prompt:
                st.error("❌ Please enter a prompt")
            else:
                with st.spinner("Generating image with Google AI..."):
                    try:
                        generated_image, used_aspect_ratio, response_text = generate_image_google(
                            prompt=prompt,
                            api_key=st.session_state.google_api_key,
                            model_key=selected_model,
                            aspect_ratio=aspect_ratio,
                            seed=seed,
                            reference_images=reference_images,
                            use_image_aspect_ratio=use_auto_aspect,
                            max_image_size=max_image_size
                        )

                        if generated_image:
                            st.success("✅ Image generated successfully!")

                            if not stealth_mode:
                                st.image(generated_image, use_container_width=True)

                            # Show response text if any
                            if response_text:
                                with st.expander("💬 Model Response Text", expanded=False):
                                    st.write(response_text)

                            st.info(f"""
                            **Generation Details:**
                            - Model: `{selected_model}`
                            - API Model ID: `{GOOGLE_IMAGE_MODELS[selected_model]}`
                            - Aspect Ratio: `{used_aspect_ratio}`
                            - Seed: `{seed if seed else 'Random'}`
                            - Reference Images: `{len(reference_images) if reference_images else 0}`
                            """)

                            if auto_save:
                                saved_path = save_image_with_metadata(
                                    generated_image, prompt,
                                    selected_model,
                                    seed if seed else random.randint(1, 1000000),
                                    used_aspect_ratio
                                )
                                st.success(f"💾 Saved to: `{saved_path}`")

                                if reference_images:
                                    comparison_img = create_comparison_image(
                                        generated_image, reference_images,
                                        prompt=prompt, model=selected_model, seed=seed
                                    )
                                    if comparison_img:
                                        comparison_path = saved_path.replace(".png", "_comparison.jpg")
                                        os.makedirs("outputs/comparisons", exist_ok=True)
                                        comparison_path = comparison_path.replace("outputs", "outputs/comparisons")
                                        comparison_img.save(comparison_path, format="JPEG", quality=75)
                                        st.success(f"💾 Comparison saved to: `{comparison_path}`")

                            # Download buttons
                            col_dl1, col_dl2 = st.columns(2)

                            with col_dl1:
                                buf = BytesIO()
                                generated_image.save(buf, format="PNG")
                                st.download_button(
                                    label="📥 Download Generated",
                                    data=buf.getvalue(),
                                    file_name=f"google_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key="google_dl_generated"
                                )

                            with col_dl2:
                                if reference_images:
                                    comparison_img = create_comparison_image(
                                        generated_image, reference_images,
                                        prompt=prompt, model=selected_model, seed=seed
                                    )
                                    if comparison_img:
                                        buf_comp = BytesIO()
                                        comparison_img.save(buf_comp, format="PNG")
                                        st.download_button(
                                            label="📥 Download Comparison",
                                            data=buf_comp.getvalue(),
                                            file_name=f"google_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                            mime="image/png",
                                            use_container_width=True,
                                            key="google_dl_comparison"
                                        )

                            with st.expander("👁️ View Comparison", expanded=False):
                                comparison_img = create_comparison_image(
                                    generated_image, reference_images,
                                    prompt=prompt, model=selected_model, seed=seed
                                )
                                if comparison_img:
                                    st.image(comparison_img, caption="Reference(s) → Generated", use_container_width=True)

                            # Add to history
                            st.session_state.google_generated_images.insert(0, {
                                'image': generated_image,
                                'prompt': prompt,
                                'model': selected_model,
                                'seed': seed,
                                'aspect_ratio': used_aspect_ratio,
                                'reference_images': reference_images,
                                'response_text': response_text,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                        else:
                            error_msg = "❌ Failed to generate image. No image data returned."
                            if response_text:
                                error_msg += f"\n\nModel response: {response_text}"
                            st.error(error_msg)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)

    # Dataset browser (imported from database_module.py)
    from database_module import render_dataset_browser
    render_dataset_browser()


    # Quick Prompt Generator Section
    st.divider()
    st.subheader("✨ Quick Prompt Generator")
    st.markdown("Generate prompts from images or text for direct use in image generation")

    with st.expander("🚀 Generate Prompt from Image/Text", expanded=False):
        qpg_col1, qpg_col2 = st.columns([1, 1])

        with qpg_col1:
            qpg_provider = st.selectbox(
                "AI Provider",
                ["OpenRouter", "Groq", "X.AI (Grok)"],
                key="google_qpg_provider"
            )

            if qpg_provider == "OpenRouter":
                from promptgen_page import OPENROUTER_MODELS
                qpg_models = OPENROUTER_MODELS
                qpg_default = "grok-4"
            elif qpg_provider == "Groq":
                from promptgen_page import GROQ_MODELS
                qpg_models = GROQ_MODELS
                qpg_default = "kimi-k2"
            else:
                from promptgen_page import XAI_MODELS
                qpg_models = XAI_MODELS
                qpg_default = "grok-4"

            qpg_model_keys = list(qpg_models.keys())
            qpg_default_idx = qpg_model_keys.index(qpg_default) if qpg_default in qpg_model_keys else 0

            qpg_model_key = st.selectbox(
                "Model", options=qpg_model_keys,
                index=qpg_default_idx, key="google_qpg_model"
            )
            qpg_model = qpg_models[qpg_model_key]

            from promptgen_page import INSTUCTIONS
            TASK_INSTRUCTIONS = INSTUCTIONS.copy()

            QPG_TASKS = {
                "GENERATE_PROMPT": "📝 Basic Prompt",
                "GENERATE_DETAILED_PROMPT": "📝 Detailed Prompt",
                "GENERATE_JSON_PROMPT": "📝 JSON Prompt"
            }
            TASK_OPTIONS = ["GENERATE_PROMPT", "GENERATE_DETAILED_PROMPT", "GENERATE_JSON_PROMPT"]
            additional_tasks = {}
            if os.path.exists("prompts/additional_tasks.json"):
                with open("prompts/additional_tasks.json", "r") as f:
                    additional_tasks = json.load(f)
                TASK_INSTRUCTIONS.update(additional_tasks)
                for key in additional_tasks.keys():
                    QPG_TASKS[key] = f"📝 {key.replace('_', ' ').title()}"
                    TASK_OPTIONS.append(key)

            qpg_task = st.selectbox(
                "Task", options=TASK_OPTIONS,
                format_func=lambda x: QPG_TASKS.get(x, x),
                key="google_qpg_task"
            )

            qpg_draft = st.text_area(
                "Draft Text (Optional)", height=100,
                placeholder="Enter draft text or description...",
                key="google_qpg_draft"
            )

            qpg_image = st.file_uploader(
                "Upload Image (Optional)",
                type=["png", "jpg", "jpeg", "webp"],
                key="google_qpg_image",
                help="Upload an image to generate prompt from"
            )

            qpg_generate = st.button("🚀 Generate Prompt", type="primary",
                                     use_container_width=True, key="google_qpg_gen_btn")

        with qpg_col2:
            st.subheader("Generated Prompt")

            if qpg_generate and (qpg_draft or qpg_image):
                try:
                    from promptgen_page import TaggerGPT, DEFAULT_SYSTEM_IMAGE_PROMPT, optimize_image

                    with st.spinner(f"Generating with {qpg_model_key}..."):
                        tagger = TaggerGPT(qpg_model)

                        instruction = TASK_INSTRUCTIONS[qpg_task]

                        if qpg_draft:
                            instruction = f"{instruction}\n\nContext/Reference text: {qpg_draft}"

                        processed_img = None
                        if qpg_image:
                            qpg_image.seek(0)
                            img = Image.open(qpg_image).convert("RGB")
                            processed_img = optimize_image(img, target_size=1120)
                            st.image(processed_img, caption=f"Processing: {qpg_image.name}", width=100)

                        result_prompt = tagger.chat_completion_prompt(
                            DEFAULT_SYSTEM_IMAGE_PROMPT,
                            instruction,
                            image=processed_img
                        )

                        st.success("✅ Prompt generated!")
                        st.session_state['google_last_generated_prompt'] = result_prompt

                        prompt_item = {
                            'result': result_prompt,
                            'task': qpg_task,
                            'model': qpg_model_key,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'has_image': qpg_image is not None,
                            'has_text': bool(qpg_draft)
                        }
                        st.session_state.google_prompt_history.insert(0, prompt_item)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

            if 'google_last_generated_prompt' in st.session_state and st.session_state['google_last_generated_prompt']:
                result_prompt = st.session_state['google_last_generated_prompt']
                prompt_hash = hash(result_prompt) % 100000

                st.text_area("Generated Result", value=result_prompt, height=200,
                             key=f"google_qpg_result_{prompt_hash}")

                if st.button("📋 Copy", key="google_copy_generated_result", use_container_width=True):
                    try:
                        import pyperclip
                        pyperclip.copy(st.session_state['google_last_generated_prompt'])
                        st.success("✅ Copied!")
                    except Exception:
                        st.code(result_prompt, language=None)
                        st.info("⚠️ Pyperclip not available. Use the code box copy button above.")

                st.download_button(
                    "💾 Download Prompt",
                    data=result_prompt,
                    file_name="generated_prompt.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="google_qpg_download"
                )
            elif not qpg_generate:
                st.info("👈 Enter text or upload an image, then click Generate")

    # Prompt History Section
    if st.session_state.google_prompt_history:
        st.divider()
        st.subheader("📜 Recent Generated Prompts")

        history_options = ["Select a recent prompt..."] + [
            f"{item['timestamp']} - {item['result'][:40]}..."
            for item in st.session_state.google_prompt_history[:10]
        ]

        hist_col1, hist_col2 = st.columns([3, 1])

        with hist_col1:
            selected_hist_idx = st.selectbox(
                "Quick access to your last 10 generated prompts",
                options=range(len(history_options)),
                format_func=lambda x: history_options[x],
                key="google_main_prompt_history",
                label_visibility="collapsed"
            )

        with hist_col2:
            if st.button("🗑️ Clear Prompt History", use_container_width=True, key="google_clear_prompt_hist"):
                st.session_state.google_prompt_history = []
                st.rerun()

        if selected_hist_idx > 0:
            hist_item = st.session_state.google_prompt_history[selected_hist_idx - 1]

            with st.expander("📝 View Prompt Details", expanded=True):
                detail_cols = st.columns([3, 1])

                with detail_cols[0]:
                    st.text_area(
                        "Prompt Content",
                        value=hist_item['result'],
                        height=150,
                        key=f"google_hist_content_{selected_hist_idx}",
                        label_visibility="collapsed"
                    )

                with detail_cols[1]:
                    st.write("**Info:**")
                    task_label = {
                        "GENERATE_PROMPT": "Basic Prompt",
                        "GENERATE_DETAILED_PROMPT": "Detailed Prompt",
                        "GENERATE_JSON_PROMPT": "JSON Prompt"
                    }.get(hist_item['task'], hist_item['task'])
                    st.caption(f"**Task:** {task_label}")
                    st.caption(f"**Model:** {hist_item['model']}")
                    st.caption(f"**Time:** {hist_item['timestamp']}")

                    source_parts = []
                    if hist_item.get('has_text'):
                        source_parts.append("Text")
                    if hist_item.get('has_image'):
                        source_parts.append("Image")
                    source = " + ".join(source_parts) if source_parts else "Unknown"
                    st.caption(f"**Source:** {source}")

                    if st.button("📋 Copy", key=f"google_copy_hist_{selected_hist_idx}", use_container_width=True):
                        try:
                            import pyperclip
                            pyperclip.copy(hist_item['result'])
                            st.success("✅ Copied!")
                        except:
                            st.info("Use code box")

                st.code(hist_item['result'], language=None)

    # Generation History
    if st.session_state.google_generated_images:
        st.divider()
        st.subheader("📜 Generation History")

        for idx, item in enumerate(st.session_state.google_generated_images[:5]):
            with st.expander(f"**{item['timestamp']}** - {item['model']}", expanded=(idx == 0)):
                cols = st.columns([1, 2])
                with cols[0]:
                    if not stealth_mode:
                        st.image(item['image'], use_container_width=True)
                    else:
                        st.info("🕶️ Hidden in Stealth Mode")
                with cols[1]:
                    st.write("**Prompt:**")
                    st.write(item['prompt'])
                    st.write(f"**Model:** {item['model']}")
                    st.write(f"**Seed:** {item.get('seed', 'N/A')}")
                    st.write(f"**Reference Images:** {len(item.get('reference_images', [])) if item.get('reference_images') else 0}")

                    if item.get('response_text'):
                        st.write(f"**Model Response:** {item['response_text'][:200]}...")

                    dl_cols = st.columns(2)

                    with dl_cols[0]:
                        buf = BytesIO()
                        metadata = PngImagePlugin.PngInfo()
                        metadata.add_text("Prompt", item['prompt'])
                        metadata.add_text("Model", item['model'])
                        metadata.add_text("Seed", str(item.get('seed', 'N/A')))
                        metadata.add_text("Timestamp", item['timestamp'])
                        metadata.add_text("Provider", "Google AI Studio")
                        item['image'].save(buf, format="PNG", pnginfo=metadata)
                        st.download_button(
                            label="📥 Download Image",
                            data=buf.getvalue(),
                            file_name=f"google_history_{idx}.png",
                            mime="image/png",
                            key=f"google_download_history_{idx}"
                        )

                    with dl_cols[1]:
                        if item.get('reference_images'):
                            comparison_img = create_comparison_image(
                                item['image'], item['reference_images'],
                                prompt=item['prompt'], model=item['model'],
                                seed=item.get('seed')
                            )
                            if comparison_img:
                                buf_comp = BytesIO()
                                comparison_img.save(buf_comp, format="PNG")
                                st.download_button(
                                    label="📥 Download Comparison",
                                    data=buf_comp.getvalue(),
                                    file_name=f"google_comparison_history_{idx}.png",
                                    mime="image/png",
                                    key=f"google_download_comparison_history_{idx}"
                                )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with Streamlit &bull; Powered by Google AI Studio API (Gemini &amp; Imagen)</p>
    </div>
    """, unsafe_allow_html=True)


# Run standalone or as imported page
if __name__ == "__main__":
    show_google_generator_page()
