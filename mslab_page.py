"""
ModelsLab Image Generator Page - Streamlit interface for ModelsLab API.

This page provides a full-featured image generation interface using the ModelsLab API:
- Text-to-Image (txt2img): Generate images from text prompts
- Image-to-Image (img2img): Transform existing images based on prompts
- Qwen Edit: Advanced image editing with Qwen model
- V7 img2img: Advanced image transformations (seedream, gen4, etc.)

Key Features:
- Asynchronous image generation with real-time status updates
- Images are displayed as they complete (polling for success status)
- Each generation call returns one image, shown immediately upon completion
- All generated images are automatically saved to outputs folder
- Support for reference images for img2img operations
- Download, copy, and save images with full metadata
- Prompt loading from YAML/JSON files
- Generation history with comparison views

The page uses the ModelsLabAPI class from modelslab/modelslab_api.py which handles:
- API requests to different endpoints (txt2img, img2img, qwen_edit, img2img_v7)
- Automatic base64 encoding for local images
- Result polling and fetching
- Image saving with proper naming

Note: ModelsLab API is asynchronous - generation requests return immediately
with a request_id, and results must be fetched via polling until status == "success".
"""

import streamlit as st
from datetime import datetime
import random
from io import BytesIO
import base64
import os
from PIL import Image, ImageOps, PngImagePlugin
import json
import yaml
import requests as http_requests
from typing import Optional, List, Dict, Any

# Import ModelsLab API
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelslab.modelslab_api import (
    ModelsLabAPI, 
    APIResponse,
    Endpoint,
    MODEL_CONFIGS,
    SIZE_IMAGE_DICT,
    QWEN_SIZE_DICT,
    SCHEDULER_LIST,
    FLUXDEV_LORAS,
    get_model_config,
    model_supports_txt2img,
    model_supports_img2img,
    decode_base64_to_image
)

# Page configuration
st.set_page_config(
    page_title="ModelsLab Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL HELPERS
# ============================================================================

DEFAULT_MODEL_BY_MODE = {
    "Text to Image": "flux-2-dev",
    "Image to Image": "flux-2-dev",
    "Qwen Edit": "qwen-edit",
}

def is_qwen_edit_model(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    config = get_model_config(model_id)
    return config.get("endpoint_img2img") == Endpoint.QWEN_EDIT


def get_models_for_mode(mode: str) -> List[str]:
    models: List[str] = []
    for model_id in MODEL_CONFIGS.keys():
        config = get_model_config(model_id)
        endpoint_img = config.get("endpoint_img2img", Endpoint.IMG2IMG)
        if mode == "Text to Image":
            if model_supports_txt2img(model_id):
                models.append(model_id)
        elif mode == "Image to Image":
            if not model_supports_img2img(model_id):
                continue
            if endpoint_img in (Endpoint.QWEN_EDIT, Endpoint.IMG2IMG_V7):
                continue
            models.append(model_id)
        elif mode == "Qwen Edit":
            if endpoint_img == Endpoint.QWEN_EDIT:
                models.append(model_id)
    return sorted(models)


def format_model_option(model_id: str) -> str:
    config = get_model_config(model_id)
    version = config.get("api_version", "").upper()
    capabilities = []
    if model_supports_txt2img(model_id):
        capabilities.append("txt2img")
    if model_supports_img2img(model_id):
        capabilities.append("img2img")
    if is_qwen_edit_model(model_id):
        capabilities.append("qwen-edit")
    if config.get("endpoint_img2img") == Endpoint.IMG2IMG_V7:
        capabilities.append("v7")
    caps_text = "/".join(capabilities) if capabilities else "n/a"
    version_text = f" · {version}" if version else ""
    return f"{model_id}{version_text} · {caps_text}"


def build_aspect_ratio_map(size_dict: Dict[str, tuple]) -> Dict[str, str]:
    return {
        f"{ratio} ({dims[0]}×{dims[1]})": ratio
        for ratio, dims in size_dict.items()
    }


def get_aspect_ratio_options(model_id: Optional[str]) -> Dict[str, str]:
    size_dict = QWEN_SIZE_DICT if is_qwen_edit_model(model_id) else SIZE_IMAGE_DICT
    return build_aspect_ratio_map(size_dict)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_api_key():
    """Load ModelsLab API key from api_keys.json or session state"""
    if 'mslab_api_key' in st.session_state and st.session_state.mslab_api_key:
        return st.session_state.mslab_api_key
    
    if os.path.exists("api_keys.json"):
        try:
            with open("api_keys.json", 'r') as f:
                api_dict = json.load(f)
                return api_dict.get("modelslab", "")
        except:
            return ""
    return ""

def get_mslab_api() -> Optional[ModelsLabAPI]:
    """Get or create ModelsLab API instance"""
    if 'mslab_api' not in st.session_state or st.session_state.mslab_api is None:
        api_key = load_api_key()
        if api_key:
            st.session_state.mslab_api = ModelsLabAPI(
                api_key=api_key,
                output_folder="outputs",
                verbose=True
            )
    return st.session_state.get('mslab_api')

def load_prompts_from_yaml(file_path="prompts.yaml"):
    root_path = "prompts/"
    file_path = os.path.join(root_path, file_path)
    """Load prompts from YAML file"""
    if not os.path.exists(file_path):
        return {}
    # if os.path.exists("prompts/prompts_custom.yaml"):
    #     file_path = "prompts/prompts_custom.yaml"
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
                return prompts_data
        return {}
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        return {}
    
def load_prompts_from_json(file_path="prompts.json"):
    root_path = "prompts/"
    file_path = os.path.join(root_path, file_path)
    """Load prompts from JSON file"""
    if not os.path.exists(file_path):
        return {}
    # if os.path.exists("prompts/prompts_custom.json"):
    #     file_path = "prompts/prompts_custom.json"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
            return prompts_data
    except Exception as e:
        st.error(f"Error loading prompts from JSON: {str(e)}")
        return {}

def flatten_json_prompts(prompts_data):
    """Flatten JSON prompts into section/category/prompt entries (like YAML UI)."""
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

        # Expect structure: Section -> Category -> PromptName -> prompt dict
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
                # Empty or non-dict category; still surface it
                flattened.append({
                    'section': section_label,
                    'category': category_key,
                    'prompt_name': category_key,
                    'prompt': dump_prompt(category_value),
                    'index': 0
                })

    # Support both legacy lowercase and new capitalized keys
    process_section("Create_Prompts", "Create")
    process_section("create_prompts", "Create")
    process_section("Edit_Prompts", "Edit")
    process_section("edit_prompts", "Edit")

    # Anything else at the top level (e.g., DATA) is treated as a standalone entry
    for key, value in prompts_data.items():
        if key in processed_sections or key.lower() in ['source', 'sources']:
            continue

        flattened.append({
            'section': 'JSON Prompts',
            'category': key,
            'prompt_name': key,
            'prompt': dump_prompt(value),
            'index': 0
        })

    return flattened

def flatten_prompts(prompts_data):
    """Flatten nested prompts dictionary into a structured list with categories"""
    flattened = []
    
    def recurse(d, path="", section=""):
        for key, value in d.items():
            current_path = f"{path} > {key}" if path else key
            if isinstance(value, list):
                for i, prompt in enumerate(value):
                    if isinstance(prompt, str) and not prompt.strip().startswith('#'):
                        flattened.append({
                            'section': section,
                            'category': current_path,
                            'prompt': prompt.strip(),
                            'index': i
                        })
            elif isinstance(value, dict):
                recurse(value, current_path, section)
    
    if 'create_prompts' in prompts_data:
        recurse(prompts_data['create_prompts'], "", "Create")
    if 'edit_prompts' in prompts_data:
        recurse(prompts_data['edit_prompts'], "", "Edit")
    
    return flattened

def encode_pil_to_base64(image: Image.Image, with_prefix: bool = True) -> str:
    """
    Encode PIL Image to base64.
    
    Args:
        image: PIL Image to encode
        with_prefix: If True, adds 'data:image/png;base64,' prefix for API compatibility
    
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if with_prefix:
        return f"data:image/png;base64,{encoded}"
    return encoded

def get_image_aspect_ratio(image: Image.Image) -> str:
    """Calculate closest aspect ratio from image dimensions"""
    width, height = image.size
    ratio = width / height
    aspect_ratios = {
        "1:1": 1.0,
        "2:3": 2/3,
        "3:2": 3/2,
        "3:4": 3/4,
        "4:3": 4/3,
        "9:16": 9/16,
        "16:9": 16/9,
    }
    closest_ratio = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
    return closest_ratio[0]

def resize_image_for_upload(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def save_image_with_metadata(image: Image.Image, prompt: str, model_name: str, 
                             seed: int, aspect_ratio: str, request_id: str = "",
                             output_folder: str = "outputs") -> str:
    """Save image with metadata to outputs folder"""
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Model", model_name)
    metadata.add_text("Seed", str(seed))
    metadata.add_text("Aspect_Ratio", aspect_ratio)
    metadata.add_text("Request_ID", str(request_id))
    metadata.add_text("Generator", "ModelsLab")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_short = prompt[:30].replace(" ", "_").replace("\n", "_")
    prompt_short = "".join(c for c in prompt_short if c.isalnum() or c in ('_', '-'))
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{prompt_short}_{model_name_short}_{timestamp}.png")
    image.save(filename, pnginfo=metadata)
    print(f"Saved image to {filename} with metadata: Prompt='{prompt[:30]}', Model='{model_name}', Seed={seed}, Aspect Ratio='{aspect_ratio}'")
    return filename

def download_image_from_response(response: APIResponse) -> Optional[Image.Image]:
    """
    Download image from API response and return as PIL Image.
    Handles both direct URLs and base64 URLs.
    """
    urls = response.output_urls or response.future_links
    if not urls or not urls[0]:
        return None
    
    url = urls[0]
    try:
        # Check if it's a base64 URL
        if ".base64" in url:
            resp = http_requests.get(url)
            resp.raise_for_status()
            img_data = decode_base64_to_image(resp.text)
        else:
            resp = http_requests.get(url)
            resp.raise_for_status()
            img_data = resp.content
        
        return Image.open(BytesIO(img_data))
    except Exception as e:
        st.error(f"Failed to download image: {e}")
        return None

def create_comparison_image(generated_image: Image.Image, reference_images: List[Image.Image], 
                           max_refs: int = 3, prompt: str = "", model: str = "", 
                           seed: Optional[int] = None) -> Optional[Image.Image]:
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
    except:
        font = ImageFont.load_default()
    
    text_y = gen_height + padding * 2 + 10
    text_x = padding
    
    metadata_text = [
        f"Model: {model}",
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

# ============================================================================
# ASYNC GENERATION MANAGEMENT
# ============================================================================

class GenerationTask:
    """Represents a pending image generation task"""
    def __init__(self, request_id: str, prompt: str, model: str, 
                 seed: int, aspect_ratio: str, reference_images: List = None):
        self.request_id = request_id
        self.prompt = prompt
        self.model = model
        self.seed = seed
        self.aspect_ratio = aspect_ratio
        self.reference_images = reference_images or []
        self.status = "processing"
        self.result_image: Optional[Image.Image] = None
        self.error_message: Optional[str] = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

def check_pending_tasks() -> Dict[str, Any]:
    """Check pending tasks once, saving finished images and returning details."""
    api = get_mslab_api()
    if not api:
        return {
            "completed": [],
            "errors": [],
            "remaining": [task.request_id for task in st.session_state.pending_tasks],
        }
    
    still_pending = []
    completed_entries: List[Dict[str, Any]] = []
    error_entries: List[Dict[str, Any]] = []
    
    for task in st.session_state.pending_tasks:
        try:
            result = api.fetch_result(task.request_id)
            
            if result.is_success:
                image = download_image_from_response(result)
                if image:
                    task.result_image = image
                    task.status = "success"
                    task.completed_at = datetime.now()
                    
                    saved_path = save_image_with_metadata(
                        image, task.prompt, task.model,
                        task.seed, task.aspect_ratio, task.request_id
                    )
                    
                    st.session_state.completed_images.insert(0, {
                        'image': image,
                        'prompt': task.prompt,
                        'model': task.model,
                        'seed': task.seed,
                        'aspect_ratio': task.aspect_ratio,
                        'request_id': task.request_id,
                        'reference_images': task.reference_images,
                        'saved_path': saved_path,
                        'timestamp': task.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    completed_entries.append({
                        "request_id": task.request_id,
                        "model": task.model,
                        "saved_path": saved_path,
                    })
                else:
                    task.status = "error"
                    task.error_message = "Failed to download image"
                    error_entries.append({
                        "request_id": task.request_id,
                        "model": task.model,
                        "error": task.error_message,
                    })
                    still_pending.append(task)
                    
            elif result.is_error:
                task.status = "error"
                task.error_message = result.error_message
                error_entries.append({
                    "request_id": task.request_id,
                    "model": task.model,
                    "error": task.error_message or "Unknown API error",
                })
                
            else:
                still_pending.append(task)
                
        except Exception as e:
            task.error_message = str(e)
            error_entries.append({
                "request_id": task.request_id,
                "model": task.model,
                "error": task.error_message,
            })
            still_pending.append(task)
    
    st.session_state.pending_tasks = still_pending
    return {
        "completed": completed_entries,
        "errors": error_entries,
        "remaining": [task.request_id for task in still_pending],
    }

def initialize_session_state():
    """Initialize all session state variables"""
    if 'mslab_api_key' not in st.session_state:
        st.session_state.mslab_api_key = load_api_key()
    if 'mslab_api' not in st.session_state:
        st.session_state.mslab_api = None
    if 'pending_tasks' not in st.session_state:
        st.session_state.pending_tasks = []  # List of GenerationTask
    if 'completed_images' not in st.session_state:
        st.session_state.completed_images = []  # List of completed generation info
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
    if 'prompts_data' not in st.session_state:
        st.session_state.prompts_data = load_prompts_from_yaml()
        st.session_state.flattened_prompts = flatten_prompts(st.session_state.prompts_data)
    if 'json_prompts_data' not in st.session_state:
        st.session_state.json_prompts_data = load_prompts_from_json()
        st.session_state.flattened_json_prompts = flatten_json_prompts(st.session_state.json_prompts_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def show_modelslab_generator_page():
    """Main function to display the ModelsLab generator page."""
    
    # Initialize session state
    initialize_session_state()

    # Main UI
    st.title("🎨 ModelsLab Image Generator")
    st.markdown("Generate images using ModelsLab API with txt2img, img2img, and Qwen edit")

    # Sidebar - Configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "ModelsLab API Key",
            value=st.session_state.mslab_api_key,
            type="password",
            help="Enter your ModelsLab API key"
        )
        if api_key_input != st.session_state.mslab_api_key:
            st.session_state.mslab_api_key = api_key_input
            st.session_state.mslab_api = None  # Reset API instance
        
        st.divider()
        
        # Generation Mode
        st.subheader("Generation Mode")
        generation_mode = st.radio(
            "Mode",
            ["Text to Image", "Image to Image", "Qwen Edit"],
            horizontal=True,
            help="Choose generation mode"
        )
        
        st.divider()
        
        # Model selection based on mode
        st.subheader("Model Settings")
        
        available_models = get_models_for_mode(generation_mode)
        if not available_models:
            st.error("No models available for this mode. Please update MODEL_CONFIGS.")
            st.stop()
        default_model = DEFAULT_MODEL_BY_MODE.get(generation_mode, available_models[0])
        if default_model not in available_models:
            default_model = available_models[0]
        
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(default_model),
            format_func=format_model_option,
            help="Choose the image generation model"
        )
        
        # LoRA selection (for flux models)
        use_lora = False
        lora_model = None
        lora_strength = 0.8
        
        if "flux" in selected_model.lower() and generation_mode != "Qwen Edit":
            st.subheader("LoRA Settings")
            use_lora = st.checkbox("Use LoRA", value=False)
            if use_lora:
                lora_options = list(FLUXDEV_LORAS.keys())
                selected_lora = st.selectbox("LoRA Model", options=lora_options)
                lora_model = FLUXDEV_LORAS[selected_lora]
                lora_strength = st.slider("LoRA Strength", 0.0, 1.0, 0.8, 0.1)
        
        # Aspect ratio
        st.subheader("Output Settings")
        
        use_auto_aspect = st.checkbox(
            "Auto-detect aspect ratio from reference image",
            value=True if generation_mode != "Text to Image" else False,
            disabled=generation_mode == "Text to Image",
            help="Automatically use the aspect ratio of the first reference image"
        )
        
        aspect_options = get_aspect_ratio_options(selected_model)
        aspect_ratio_display = st.selectbox(
            "Aspect Ratio",
            options=list(aspect_options.keys()),
            index=0,
            disabled=use_auto_aspect,
            help="Target aspect ratio for generated image"
        )
        aspect_ratio = aspect_options.get(aspect_ratio_display)
        
        # Prompt source settings
        PROMPTS_FILES = {
            "base": "prompts",
            "custom": "prompts_custom",
            "assets": "prompts_assets"
        }
        default_prompts = "custom" if (
            os.path.exists("prompts/prompts_custom.json") or 
            os.path.exists("prompts/prompts_custom.yaml")
        ) else "base"

        st.subheader("Prompt Source Settings")
        selected_prompts = st.selectbox(
            "Prompts Source",
            options=list(PROMPTS_FILES.keys()),
            index=list(PROMPTS_FILES.keys()).index(default_prompts),
            help="Choose the prompt file to use"
        )

        # Generation parameters
        st.subheader("Generation Parameters")
        
        use_random_seed = st.checkbox("Use random seed", value=True)
        if use_random_seed:
            seed = random.randint(1, 1000000)
        else:
            seed = st.number_input("Seed", min_value=1, max_value=1000000, value=12345)
        
        # Model-specific parameters
        model_config = get_model_config(selected_model)
        
        if generation_mode != "Qwen Edit":
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=4,
                max_value=50,
                value=model_config.get("num_inference_steps", 20),
                help="More steps = better quality but slower"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="How closely to follow the prompt"
            )
            
            if generation_mode == "Image to Image":
                strength = st.slider(
                    "Transformation Strength",
                    min_value=0.1,
                    max_value=1.0,
                    value=model_config.get("strength", 0.7),
                    step=0.1,
                    help="How much to transform the input image"
                )
            else:
                strength = 0.7
            
            scheduler = st.selectbox(
                "Scheduler",
                options=SCHEDULER_LIST,
                index=SCHEDULER_LIST.index("DPMSolverMultistepScheduler"),
                help="Sampling scheduler algorithm"
            )
        else:
            num_inference_steps = 8
            guidance_scale = 7.5
            strength = 0.7
            scheduler = "DPMSolverMultistepScheduler"
        
        # Reference image settings
        if generation_mode != "Text to Image":
            st.subheader("Reference Image Settings")
            max_image_size = st.slider(
                "Max reference image size (px)",
                min_value=512,
                max_value=2048,
                value=1024,
                step=128,
                help="Maximum dimension for reference images"
            )
            
            default_resize_mp = 1.7 if is_qwen_edit_model(selected_model) else 1.0
            resize_mp = st.slider(
                "Resize to Megapixels",
                min_value=0.5,
                max_value=2.0,
                value=default_resize_mp,
                step=0.1,
                help="Target megapixels for input image"
            )
        else:
            max_image_size = 1024
            resize_mp = 1.0
        
        st.divider()
        stealth_mode = st.checkbox("🕶️ Stealth Mode", value=False, help="Hide all image thumbnails")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        # header less the subheader
        st.subheader("📝 Prompt & Reference Images", ) 
        
        # Prompt source selector
        prompt_source = st.radio(
            "Prompt Source",
            options=["Custom Prompt", "Load from YAML", "Load from JSON"],
            horizontal=True,
            help="Choose to write your own prompt or load from YAML/JSON files"
        )
        
        prompt = ""
        
        if prompt_source == "Load from JSON":
            # reload ech time st.session_state.flattened_json_prompts
            json_prompts_data = load_prompts_from_json(PROMPTS_FILES[selected_prompts] + ".json")
            st.session_state.json_prompts_data = json_prompts_data
            st.session_state.flattened_json_prompts = flatten_json_prompts(json_prompts_data)

            
            if st.session_state.flattened_json_prompts:
                # Mirror YAML selection: Section -> Category -> Prompt
                sections = {}
                for item in st.session_state.flattened_json_prompts:
                    section = item.get('section', 'JSON Prompts') or 'JSON Prompts'
                    sections.setdefault(section, []).append(item)

                col_section, col_category = st.columns([1, 2])

                with col_section:
                    section_options = list(sections.keys())
                    selected_section = st.selectbox(
                        "Section",
                        options=section_options,
                        help="Choose between Create or Edit prompts",
                        key="json_section_select"
                    )

                section_prompts = sections[selected_section]
                categories = sorted(list(set([p['category'] for p in section_prompts])))

                with col_category:
                    selected_category = st.selectbox(
                        "Category",
                        options=categories,
                        help="Choose a category of prompts",
                        key="json_category_select"
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
                    key="json_prompt_select"
                )

                # Manage prompt text state to allow editing while updating on selection change
                selection_key = f"{selected_section}_{selected_category}_{selected_prompt_idx}"
                
                # Check if selection has changed
                if 'last_json_selection' not in st.session_state or st.session_state.last_json_selection != selection_key:
                    st.session_state.last_json_selection = selection_key
                    if selected_prompt_idx > 0:
                        # Update text with selected prompt
                        st.session_state['json_prompt_input'] = filtered_prompts[selected_prompt_idx - 1]['prompt']
                    else:
                        # Clear text if no prompt selected
                        st.session_state['json_prompt_input'] = ""
                
                # Initialize if not exists
                if 'json_prompt_input' not in st.session_state:
                     st.session_state['json_prompt_input'] = ""

                prompt = st.text_area(
                    "Selected Prompt (editable)" if selected_prompt_idx > 0 else "Image Prompt",
                    key="json_prompt_input",
                    height=200,
                    placeholder="Select a prompt from the dropdowns above or type here...",
                    help="You can edit the loaded prompt before generating"
                )
            else:
                st.warning("⚠️ No prompts found in prompts.json")
                prompt = st.text_area(
                    "Image Prompt",
                    height=200,
                    placeholder="Describe the image you want to generate...",
                    help="Enter a detailed description of the image you want to create",
                    key="custom_prompt_fallback_json"
                )
        
        elif prompt_source == "Load from YAML":
            prompts_data = load_prompts_from_yaml(f"{PROMPTS_FILES[selected_prompts]}.yaml")
            st.session_state.prompts_data = prompts_data

            if st.session_state.prompts_data:
                prompts_data = st.session_state.prompts_data
                
                # Build hierarchical structure: Section -> Category -> Sublevel -> Prompts
                # Section mapping
                section_keys = {
                    "Create": "create_prompts",
                    "Edit": "edit_prompts"
                }
                available_sections = [s for s, k in section_keys.items() if k in prompts_data]
                
                if available_sections:
                    # Create a hierarchical selector with 3 levels
                    col_section, col_category = st.columns([1, 1])
                    
                    with col_section:
                        selected_section = st.selectbox(
                            "Section",
                            options=available_sections,
                            help="Choose between Create or Edit prompts",
                            key="yaml_section_select"
                        )
                    
                    # Get categories for selected section
                    section_key = section_keys[selected_section]
                    section_data = prompts_data.get(section_key, {})
                    categories = list(section_data.keys())
                    
                    with col_category:
                        selected_category = st.selectbox(
                            "Category",
                            options=categories if categories else ["No categories"],
                            help="Choose a category of prompts",
                            key="yaml_category_select"
                        )
                    
                    # Get sublevels for selected category
                    category_data = section_data.get(selected_category, {})
                    
                    # Determine if category_data has sublevels (dict) or is directly a list of prompts
                    if isinstance(category_data, dict):
                        sublevels = list(category_data.keys())
                        has_sublevels = True
                    elif isinstance(category_data, list):
                        sublevels = ["(prompts)"]
                        has_sublevels = False
                    else:
                        sublevels = []
                        has_sublevels = False
                    
                    # Sublevel selector (below section/category)
                    if has_sublevels and sublevels:
                        selected_sublevel = st.selectbox(
                            "Sublevel",
                            options=sublevels,
                            help="Choose a sublevel of prompts",
                            key="yaml_sublevel_select"
                        )
                    else:
                        selected_sublevel = None
                    
                    # Get prompts based on selection
                    if has_sublevels and selected_sublevel:
                        sublevel_data = category_data.get(selected_sublevel, [])
                        # Check if sublevel_data is another dict (4th level nesting) or a list
                        if isinstance(sublevel_data, dict):
                            # Handle 4th level: flatten it
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
                        # No sublevels, category_data is the list of prompts
                        if isinstance(category_data, list):
                            prompts_list = [p.strip() for p in category_data if isinstance(p, str) and not p.strip().startswith('#')]
                        else:
                            prompts_list = []
                    
                    # Create options for selectbox with preview
                    prompt_options = ["Select a prompt..."] + [
                        f"{p[:80]}..." if len(p) > 80 else p
                        for p in prompts_list
                    ]
                    
                    selected_prompt_idx = st.selectbox(
                        "Select Prompt",
                        options=range(len(prompt_options)),
                        format_func=lambda x: prompt_options[x],
                        help="Choose a specific prompt",
                        key="yaml_prompt_select"
                    )
                    
                    if selected_prompt_idx > 0:
                        selected_prompt_text = prompts_list[selected_prompt_idx - 1]
                        # Use dynamic key to force refresh when selection changes
                        sublevel_key = selected_sublevel if selected_sublevel else "none"
                        prompt = st.text_area(
                            "Selected Prompt (editable)",
                            value=selected_prompt_text,
                            height=150,
                            key=f"yaml_prompt_{selected_section}_{selected_category}_{sublevel_key}_{selected_prompt_idx}",
                            help="You can edit the loaded prompt before generating"
                        )
                    else:
                        prompt = st.text_area(
                            "Image Prompt",
                            height=150,
                            placeholder="Select a prompt from the dropdowns above...",
                            help="Select section, category, sublevel and prompt from the dropdowns",
                            key="empty_yaml_prompt"
                        )
                else:
                    st.warning("⚠️ No valid sections found in prompts.yaml")
                    prompt = st.text_area(
                        "Image Prompt",
                        height=150,
                        placeholder="Describe the image you want to generate...",
                        help="Enter a detailed description of the image you want to create",
                        key="custom_prompt_fallback_yaml"
                    )
            else:
                st.warning("⚠️ No prompts found in prompts.yaml")
                prompt = st.text_area(
                    "Image Prompt",
                    height=150,
                    placeholder="Describe the image you want to generate...",
                    help="Enter a detailed description of the image you want to create",
                    key="custom_prompt_fallback"
                )
        else:
            # Custom prompt input
            prompt = st.text_area(
                "Image Prompt",
                height=150,
                placeholder="Describe the image you want to generate...",
                help="Enter a detailed description of the image you want to create",
                key="custom_prompt"
            )
        
        # Reference images upload (for img2img modes)
        reference_images = None
        if generation_mode != "Text to Image":
            st.subheader("Reference Images")
            uploaded_files = st.file_uploader(
                "Upload reference images",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                accept_multiple_files=True,
                help="Upload one or more reference images"
            )
            
            if uploaded_files:
                max_refs = 4 if generation_mode == "Qwen Edit" else len(uploaded_files)
                if generation_mode == "Qwen Edit" and len(uploaded_files) > max_refs:
                    st.info("Qwen Edit supports up to 4 reference images. Using the first 4 files.")
                used_files = uploaded_files[:max_refs]
                st.write(f"**{len(used_files)} image(s) will be used**")
                num_cols = max(1, min(len(used_files), 3))
                ref_cols = st.columns(num_cols)
                reference_images = []
                
                for idx, uploaded_file in enumerate(used_files):
                    img = Image.open(uploaded_file)
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    reference_images.append(img)
                    
                    with ref_cols[idx % num_cols]:
                        if not stealth_mode:
                            st.image(img, caption=f"Ref {idx+1}", width=150)
                        st.caption(f"Size: {img.size[0]}×{img.size[1]}")
        
        # Generate button
        can_generate = prompt and st.session_state.mslab_api_key
        if generation_mode != "Text to Image":
            can_generate = can_generate and reference_images
        
        generate_btn = st.button(
            "🎨 Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not can_generate
        )

    with col2:
        st.subheader("🖼️ Generated Images")
        fetch_results_btn = st.button(
            "🔄 Fetch Pending Results",
            use_container_width=True,
            disabled=not bool(st.session_state.pending_tasks),
            help="Run a single status check for all pending requests"
        )
        if fetch_results_btn:
            with st.spinner("Fetching latest results and saving outputs..."):
                stats = check_pending_tasks()
            if stats["completed"]:
                saved_lines = [
                    f"• {entry['request_id']} → saved to {entry['saved_path']}"
                    for entry in stats["completed"]
                ]
                st.success(
                    "\n".join([
                        f"Fetched {len(stats['completed'])} completed generation(s):",
                        *saved_lines,
                        "All images were saved with metadata in outputs/.",
                    ])
                )
            if stats["errors"]:
                error_lines = [
                    f"• {entry['request_id']} ({entry['model']}): {entry['error']}"
                    for entry in stats["errors"]
                ]
                st.warning(
                    "\n".join([
                        f"{len(stats['errors'])} request(s) returned errors:",
                        *error_lines,
                    ])
                )
            if stats["remaining"]:
                st.info(
                    "\n".join([
                        f"Still processing {len(stats['remaining'])} request(s):",
                        *[f"• {req_id}" for req_id in stats["remaining"]],
                        "Press fetch again later to refresh.",
                    ])
                )
            if not stats["completed"] and not stats["errors"] and not stats["remaining"]:
                st.info("No pending requests right now. Submit a new generation to begin.")
        
        # Show pending tasks status
        if st.session_state.pending_tasks:
            st.info(f"⏳ {len(st.session_state.pending_tasks)} generation(s) in progress...")
            
            for task in st.session_state.pending_tasks:
                elapsed = (datetime.now() - task.created_at).seconds
                req_id = str(task.request_id)[:8] if task.request_id else "unknown"
                st.caption(f"Request `{req_id}...` - {task.model} - {elapsed}s elapsed")
        
        # Handle generation
        if generate_btn:
            if not st.session_state.mslab_api_key:
                st.error("❌ Please enter your ModelsLab API key")
            elif not prompt:
                st.error("❌ Please enter a prompt")
            elif generation_mode != "Text to Image" and not reference_images:
                st.error("❌ Please upload reference image(s)")
            else:
                api = get_mslab_api()
                if not api:
                    st.session_state.mslab_api = ModelsLabAPI(
                        api_key=st.session_state.mslab_api_key,
                        output_folder="outputs",
                        verbose=True
                    )
                    api = st.session_state.mslab_api
                
                with st.spinner("Submitting generation request..."):
                    try:
                        # Determine actual aspect ratio
                        if use_auto_aspect and reference_images:
                            actual_aspect_ratio = get_image_aspect_ratio(reference_images[0])
                        else:
                            actual_aspect_ratio = aspect_ratio
                        
                        # Prepare images for API
                        image_data = None
                        if reference_images:
                            # Resize and encode to base64
                            resized_imgs = [resize_image_for_upload(img, max_image_size) for img in reference_images]
                            image_data = [encode_pil_to_base64(img) for img in resized_imgs]
                        
                        images_payload = image_data if reference_images else None

                        generation_kwargs: Dict[str, Any] = {
                            "aspect_ratio": actual_aspect_ratio,
                            "seed": seed,
                        }

                        if generation_mode == "Text to Image":
                            generation_kwargs.update({
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "scheduler": scheduler,
                                "lora_model": lora_model if use_lora else None,
                                "lora_strength": lora_strength if use_lora else None,
                            })
                        elif generation_mode == "Image to Image":
                            generation_kwargs.update({
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "scheduler": scheduler,
                                "strength": strength,
                                "resize_mp": resize_mp,
                                "lora_model": lora_model if use_lora else None,
                                "lora_strength": lora_strength if use_lora else None,
                            })
      
                        else:  # Qwen Edit
                            generation_kwargs.update({
                                "resize_mp": resize_mp,
                                "num_inference_steps": num_inference_steps,
                            })

                        if selected_model == "flux-2-dev":
                            print(f"Using flux-2-dev specific parameters: strength = {strength}")
                            generation_kwargs = {
                                "aspect_ratio": actual_aspect_ratio,
                                "seed": seed,
                                # "guidance_scale": guidance_scale,
                                # "scheduler": scheduler,
                                "strength": strength,
                                # "lora_model": lora_model if use_lora else None,
                                # "lora_strength": lora_strength if use_lora else None,
                            }
                            #  param flux-2-dev      {
                            #     "key": "...",
                            #     "deploy_type": "flux_2_dev",
                            #     "prompt": "...",
                            #     "negative_prompt": "...",
                            #     "init_image": ["url_immagine"],
                            #     "width": "1024",
                            #     "height": "1024",
                            #     "samples": "1",
                            #     "strength": 0.7,  # ← questo è l'unico param di controllo img2img confermato
                            #     "seed": null,
                            #     "webhook": null,
                            #     "track_id": null
                            # }

                        response = api.generate(
                            prompt=prompt,
                            images=images_payload,
                            model_id=selected_model,
                            **generation_kwargs,
                        )
                        
                        if response.is_error:
                            st.error(f"❌ API Error: {response.error_message}")
                        else:
                            # Add to pending tasks
                            task = GenerationTask(
                                request_id=response.request_id,
                                prompt=prompt,
                                model=selected_model,
                                seed=seed,
                                aspect_ratio=actual_aspect_ratio,
                                reference_images=reference_images
                            )
                            st.session_state.pending_tasks.append(task)
                            st.success(f"✅ Request submitted! ID: `{response.request_id}`")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        


        # Display completed images
        if st.session_state.completed_images:
            for idx, item in enumerate(st.session_state.completed_images[:5]):
                with st.expander(f"**{item['timestamp']}** - {item['model']}", expanded=(idx == 0)):
                    if not stealth_mode:
                        st.image(item['image'], use_container_width=True)
                    
                    req_id_display = str(item['request_id'])[:12] if item.get('request_id') else 'N/A'
                    st.info(f"""
                    **Details:**
                    - Model: `{item['model']}`
                    - Aspect Ratio: `{item['aspect_ratio']}`
                    - Seed: `{item['seed']}`
                    - Request ID: `{req_id_display}...`
                    - Saved to: `{item.get('saved_path', 'N/A')}`
                    """)
                    
                    # Action buttons
                    btn_cols = st.columns(3)
                    
                    with btn_cols[0]:
                        buf = BytesIO()
                        metadata = PngImagePlugin.PngInfo()
                        metadata.add_text("Prompt", item['prompt'])
                        metadata.add_text("Model", item['model'])
                        metadata.add_text("Seed", str(item['seed']))
                        item['image'].save(buf, format="PNG", pnginfo=metadata)
                        dl_req_id = str(item['request_id'])[:8] if item.get('request_id') else 'img'
                        st.download_button(
                            "📥 Download",
                            data=buf.getvalue(),
                            file_name=f"mslab_{dl_req_id}.png",
                            mime="image/png",
                            key=f"dl_{idx}"
                        )
                    
                    with btn_cols[1]:
                        if st.button("📋 Copy Prompt", key=f"copy_{idx}"):
                            try:
                                import pyperclip
                                pyperclip.copy(item['prompt'])
                                st.success("✅ Copied!")
                            except:
                                st.code(item['prompt'])
                    
                    with btn_cols[2]:
                        if item.get('reference_images'):
                            comparison = create_comparison_image(
                                item['image'],
                                item['reference_images'],
                                prompt=item['prompt'],
                                model=item['model'],
                                seed=item['seed']
                            )
                            if comparison:
                                buf_comp = BytesIO()
                                comparison.save(buf_comp, format="PNG")
                                comp_req_id = str(item['request_id'])[:8] if item.get('request_id') else 'comp'
                                st.download_button(
                                    "📥 Comparison",
                                    data=buf_comp.getvalue(),
                                    file_name=f"comparison_{comp_req_id}.png",
                                    mime="image/png",
                                    key=f"comp_{idx}"
                                )

    
    # Dataset browser (imported from database_module.py)
    from database_module import render_dataset_browser
    render_dataset_browser()

    """
    # ho dovuto commentare tutto perchè, dopo la prima generazione lapp si blocca e non genera piu nulla, anche l altre pagine si bloccano
    
    # Quick Prompt Generator Section
    st.divider()
    st.subheader("✨ Quick Prompt Generator")
    st.markdown("Generate prompts from images or text for direct use in image generation")

    with st.expander("🚀 Generate Prompt from Image/Text", expanded=False):
        qpg_col1, qpg_col2 = st.columns([1, 1])
        
        with qpg_col1:
            # Model selection for prompt generation
            qpg_provider = st.selectbox(
                "AI Provider",
                ["OpenRouter", "Groq", "X.AI (Grok)"],
                key="qpg_provider"
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
                "Model",
                options=qpg_model_keys,
                index=qpg_default_idx,
                key="qpg_model"
            )
            qpg_model = qpg_models[qpg_model_key]
            
            # Task selection
            from promptgen_page import INSTUCTIONS
            TASK_INSTRUCTIONS = INSTUCTIONS.copy()

            # if the file prompts/additional_tasks.json exists, load additional instructions
            # Add them all in INSTRUCTIONS
            import json
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
                "Task",
                options=TASK_OPTIONS,
                format_func=lambda x: QPG_TASKS.get(x, x),
                key="qpg_task"
            )
            
            # Draft text input
            qpg_draft = st.text_area(
                "Draft Text (Optional)",
                height=100,
                placeholder="Enter draft text or description...",
                key="qpg_draft"
            )
            
            # Image upload (no preview)
            qpg_image = st.file_uploader(
                "Upload Image (Optional)",
                type=["png", "jpg", "jpeg", "webp"],
                key="qpg_image",
                help="Upload an image to generate prompt from"
            )
            
            qpg_generate = st.button("🚀 Generate Prompt", type="primary", width="stretch", key="qpg_gen_btn")
        
        with qpg_col2:
            st.subheader("Generated Prompt")
            
            if qpg_generate and (qpg_draft or qpg_image):
                try:
                    from promptgen_page import TaggerGPT, DEFAULT_SYSTEM_IMAGE_PROMPT, optimize_image
                    
                    with st.spinner(f"Generating with {qpg_model_key}..."):
                        tagger = TaggerGPT(qpg_model)
                        
                        # Build instruction
                        instruction = TASK_INSTRUCTIONS[qpg_task]
                        
                        if qpg_draft:
                            instruction = f"{instruction}\n\nContext/Reference text: {qpg_draft}"
                        
                        # Process image if provided
                        processed_img = None
                        if qpg_image:
                            qpg_image.seek(0)  # Reset file pointer to beginning
                            img = Image.open(qpg_image).convert("RGB")
                            processed_img = optimize_image(img, target_size=1120)
                            # Debug: show processed image size
                            st.image(processed_img, caption=f"Processing: {qpg_image.name}", width=100)

                        # Generate
                        result_prompt = tagger.chat_completion_prompt(
                            DEFAULT_SYSTEM_IMAGE_PROMPT,
                            instruction,
                            image=processed_img
                        )
                        
                        st.success("✅ Prompt generated!")
                        
                        # Save to session state for immediate access
                        st.session_state['last_generated_prompt'] = result_prompt
                        
                        # Save to history
                        prompt_item = {
                            'result': result_prompt,
                            'task': qpg_task,
                            'model': qpg_model_key,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'has_image': qpg_image is not None,
                            'has_text': bool(qpg_draft)
                        }
                        st.session_state.prompt_history.insert(0, prompt_item)
                        

                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Display result if available (outside the generate block so it persists)
            if 'last_generated_prompt' in st.session_state and st.session_state['last_generated_prompt']:
                result_prompt = st.session_state['last_generated_prompt']
                
                # Use dynamic key based on prompt content hash to force refresh
                prompt_hash = hash(result_prompt) % 100000
                
                # Display result in text area (editable)
                st.text_area("Generated Result", value=result_prompt, height=200, key=f"qpg_result_{prompt_hash}")

                if st.button("📋 Copy", key="copy_generated_result", width="stretch"):
                    try:
                        import pyperclip
                        pyperclip.copy(st.session_state['last_generated_prompt'])
                        st.success("✅ Copied!")
                    except Exception as e:
                        # Display result in a code block with built-in copy button
                        st.code(result_prompt, language=None)
                        st.info("⚠️ Pyperclip not available. Use the code box copy button above.")
                
                # Download button
                st.download_button(
                    "💾 Download Prompt",
                    data=result_prompt,
                    file_name="generated_prompt.txt",
                    mime="text/plain",
                    width="stretch",
                    key="qpg_download"
                )
            elif not qpg_generate:
                st.info("👈 Enter text or upload an image, then click Generate")

    # Prompt History Section
    if st.session_state.prompt_history:
        st.divider()
        st.subheader("📜 Recent Generated Prompts")
        
        # Create dropdown options
        history_options = ["Select a recent prompt..."] + [
            f"{item['timestamp']} - {item['result'][:40]}..."
            for item in st.session_state.prompt_history[:10]
        ]
        
        hist_col1, hist_col2 = st.columns([3, 1])
        
        with hist_col1:
            selected_hist_idx = st.selectbox(
                "Quick access to your last 10 generated prompts",
                options=range(len(history_options)),
                format_func=lambda x: history_options[x],
                key="main_prompt_history",
                label_visibility="collapsed"
            )
        
        with hist_col2:
            if st.button("🗑️ Clear Prompt History", width="stretch"):
                st.session_state.prompt_history = []
                st.rerun()
        
        if selected_hist_idx > 0:
            hist_item = st.session_state.prompt_history[selected_hist_idx - 1]
            
            with st.expander("📝 View Prompt Details", expanded=True):
                detail_cols = st.columns([3, 1])
                
                with detail_cols[0]:
                    st.text_area(
                        "Prompt Content",
                        value=hist_item['result'],
                        height=150,
                        key=f"hist_content_{selected_hist_idx}",
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
                    
                    # Copy button
                    if st.button("📋 Copy", key=f"copy_hist_{selected_hist_idx}", width="stretch"):
                        try:
                            import pyperclip
                            pyperclip.copy(hist_item['result'])
                            st.success("✅ Copied!")
                        except:
                            st.info("Use code box")
                
                # Code block for easy copying
                st.code(hist_item['result'], language=None)
"""

    # ============================================================================
    # FOOTER
    # ============================================================================

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with Streamlit • Powered by ModelsLab API</p>
    </div>
    """, unsafe_allow_html=True)