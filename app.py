from pyperclip import copy
import streamlit as st
from openai import OpenAI
from datetime import datetime
import random
from io import BytesIO
import base64
import os
from PIL import Image, PngImagePlugin
import json
import yaml

# Page configuration
st.set_page_config(
    page_title="Image Tools App",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model definitions
OPENROUTER_IMAGE_MODELS = {
    "flux.2-klein-4b":"black-forest-labs/flux.2-klein-4b",
    "flux.2-pro": "black-forest-labs/flux.2-pro",
    "flux.2-flex": "black-forest-labs/flux.2-flex",
    "flux.2-max": "black-forest-labs/flux.2-max",
    "gemini-2.5-flash-image": "google/gemini-2.5-flash-image",
    "gemini-3-pro-image-preview": "google/gemini-3-pro-image-preview",
    "gpt-5-image-mini": "openai/gpt-5-image-mini",
    "gpt-5-image": "openai/gpt-5-image",

    "riverflow-v2-fast-preview" :"sourceful/riverflow-v2-fast-preview",
    "riverflow-v2-standard-preview" :"sourceful/riverflow-v2-standard-preview",
    "riverflow-v2-max-preview" :"sourceful/riverflow-v2-max-preview",

    "seedream-4.5": "bytedance-seed/seedream-4.5",
}
default_model = "gemini-2.5-flash-image"

ASPECT_RATIOS = {
    "1:1 (1024×1024)": "1:1",
    "2:3 (832×1248)": "2:3",
    "3:2 (1248×832)": "3:2",
    "3:4 (864×1184)": "3:4",
    "4:3 (1184×864)": "4:3",
    "4:5 (896×1152)": "4:5",
    "5:4 (1152×896)": "5:4",
    "9:16 (768×1344)": "9:16",
    "16:9 (1344×768)": "16:9",
    "21:9 (1536×672)": "21:9"
}

# Helper functions
def load_api_key():
    """Load API key from api_keys.json or session state"""
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    
    if os.path.exists("api_keys.json"):
        try:
            with open("api_keys.json", 'r') as f:
                api_dict = json.load(f)
                return api_dict.get("openrouter", "")
        except:
            return ""
    return ""

def load_prompts_from_yaml(file_path="prompts.yaml"):
    """Load prompts from YAML file"""
    if not os.path.exists(file_path):
        return {}
    if os.path.exists("prompts_custom.yaml"):
        file_path = "prompts_custom.yaml"
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
    """Load prompts from JSON file"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
            return prompts_data
    except Exception as e:
        st.error(f"Error loading prompts from JSON: {str(e)}")
        return {}

def flatten_json_prompts(prompts_data):
    """Flatten JSON prompts into a structured list"""
    flattened = []
    
    for key, value in prompts_data.items():
        # Skip 'source' and other metadata keys
        if key in ['source', 'sources']:
            continue
        
        if isinstance(value, dict):
            # Convert the entire dict to a readable prompt string
            prompt_text = json.dumps(value, indent=2, ensure_ascii=False)
            flattened.append({
                'section': 'JSON Prompts',
                'category': key,
                'prompt': prompt_text,
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
                # Found actual prompts
                for i, prompt in enumerate(value):
                    # Clean up comments (lines starting with #)
                    if isinstance(prompt, str) and not prompt.strip().startswith('#'):
                        flattened.append({
                            'section': section,
                            'category': current_path,
                            'prompt': prompt.strip(),
                            'index': i
                        })
            elif isinstance(value, dict):
                # Keep recursing
                recurse(value, current_path, section)
    
    # Process create_prompts
    if 'create_prompts' in prompts_data:
        recurse(prompts_data['create_prompts'], "", "Create")
    
    # Process edit_prompts
    if 'edit_prompts' in prompts_data:
        recurse(prompts_data['edit_prompts'], "", "Edit")
    
    return flattened

def get_client(api_key):
    """Initialize OpenRouter client"""
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

def encode_image_to_base64(image):
    """Encode PIL Image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_data

def get_image_aspect_ratio(image):
    """Calculate closest aspect ratio from image dimensions"""
    width, height = image.size
    ratio = width / height
    aspect_ratios = {
        "1:1": 1.0,
        "2:3": 2/3,
        "3:2": 3/2,
        "3:4": 3/4,
        "4:3": 4/3,
        "4:5": 4/5,
        "5:4": 5/4,
        "9:16": 9/16,
        "16:9": 16/9,
        "21:9": 21/9
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

def generate_image(prompt, api_key, model_name, aspect_ratio, seed, reference_images=None, 
                   use_image_aspect_ratio=False, max_image_size=1024):
    """Generate image using OpenRouter API"""
    
    if not seed:
        seed = random.randint(1, 1000000)
    
    client = get_client(api_key)
    
    # Build message content
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }
    
    # Process reference images if provided
    if reference_images:
        # Use first image aspect ratio if enabled
        if use_image_aspect_ratio and len(reference_images) > 0:
            aspect_ratio = get_image_aspect_ratio(reference_images[0])
        
        for img in reference_images:
            # Resize image if needed
            resized_img = resize_image(img, max_size=max_image_size)
            image_data = encode_image_to_base64(resized_img)
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })
    
    # Make API call
    PARAM = {
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": aspect_ratio}
        }
    if model_name in ["google/gemini-2.5-flash-image", "google/gemini-3-pro-image-preview"]:
        PARAM["image_config"]["image_size"] = "2K"
    print("Generating with params:", PARAM)

    response_full = client.chat.completions.create(
        model=model_name,
        messages=[user_message],
        seed=seed,
        extra_body=PARAM
    )
    
    # Extract generated image
    response = response_full.choices[0].message
    if response.images:
        base64_data = response.images[0]['image_url']['url']
        header, encoded = base64_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        return image, aspect_ratio
    
    return None, aspect_ratio

def save_image_with_metadata(image, prompt, model_name, seed, aspect_ratio,
                              output_folder="outputs", reduce_quality=False):
    """Save image with metadata to outputs folder"""
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Model", model_name)
    metadata.add_text("Seed", str(seed))
    metadata.add_text("Aspect_Ratio", aspect_ratio)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_short = prompt[:30].replace(" ", "_").replace("\n", "_")
    # replce all special characters in prompt_short that can interfere with file saving
    prompt_short = "".join(c for c in prompt_short if c.isalnum() or c in ('_', '-'))

    model_name_short = model_name.split("/")[-1]
    
    os.makedirs(output_folder, exist_ok=True)
    
    filename = os.path.join(output_folder, f"{prompt_short}_{model_name_short}_{timestamp}.png")
    image.save(filename, pnginfo=metadata)
    return filename

def create_comparison_image(generated_image, reference_images, max_refs=3, prompt="", model="", seed=None):
    """Create a side-by-side comparison image with reference and generated images, including generation parameters"""
    if not reference_images:
        return None
    
    from PIL import ImageDraw, ImageFont
    
    # Limit number of reference images
    ref_images = reference_images[:max_refs]
    
    # Calculate dimensions
    gen_width, gen_height = generated_image.size
    
    # Resize reference images to match generated image height
    resized_refs = []
    total_ref_width = 0
    for ref_img in ref_images:
        aspect = ref_img.width / ref_img.height
        new_height = gen_height
        new_width = int(new_height * aspect)
        resized_ref = ref_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_refs.append(resized_ref)
        total_ref_width += new_width
    
    # Add padding
    padding = 20
    text_area_height = 100  # Space for metadata text at bottom
    total_width = total_ref_width + gen_width + padding * (len(resized_refs) + 1)
    
    # Create comparison canvas with extra space for text
    comparison = Image.new('RGB', (total_width, gen_height + padding * 2 + text_area_height), color='white')
    
    # Paste reference images
    x_offset = padding
    for ref_img in resized_refs:
        comparison.paste(ref_img, (x_offset, padding))
        x_offset += ref_img.width + padding
    
    # Paste generated image
    comparison.paste(generated_image, (x_offset, padding))
    
    # Add metadata text at the bottom
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    
    # Position for text
    text_y = gen_height + padding * 2 + 10
    text_x = padding
    
    # Draw metadata
    metadata_text = [
        f"Model: {model}",
        f"Seed: {seed if seed else 'Random'}"
    ]
    
    for i, line in enumerate(metadata_text):
        draw.text((text_x, text_y + i * 20), line, fill='black', font=font)
    
    # Draw prompt (wrapped if too long)
    prompt_y = text_y + 45
    prompt_text = f"Prompt: {prompt}"
    if len(prompt_text) > 150:
        prompt_text = prompt_text[:150] + "..."
    draw.text((text_x, prompt_y), prompt_text, fill='black', font=font)
    
    return comparison

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = load_api_key()
if 'prompts_data' not in st.session_state:
    prompts_data = load_prompts_from_yaml()
    st.session_state.prompts_data = prompts_data
    st.session_state.flattened_prompts = flatten_prompts(prompts_data)
if 'json_prompts_data' not in st.session_state:
    json_prompts_data = load_prompts_from_json()
    st.session_state.json_prompts_data = json_prompts_data
    st.session_state.flattened_json_prompts = flatten_json_prompts(json_prompts_data)

# Page Navigation
with st.sidebar:
    st.title("🎨 Image Tools")
    page = st.radio(
        "Navigate",
        ["Image Generator", "Prompt Generator", "Image Viewer", "Prompt Manager"],
        label_visibility="collapsed"
    )
    st.divider()

if page == "Image Viewer":
    # Import and run the image viewer page
    import image_viewer_page
    image_viewer_page.show_image_viewer_page()
    st.stop()

if page == "Prompt Manager":
    # Import and run the prompt manager page
    import prompt_manager_page
    prompt_manager_page.show_prompt_manager_page()
    st.stop()

if page == "Prompt Generator":
    # Import and run the tagger page
    import promptgen_page
    promptgen_page.show_tagger_page()
    st.stop()

# Main UI
st.title("🎨 OpenRouter Image Generator")
st.markdown("Generate images using OpenRouter API with customizable parameters")

# Sidebar - Configuration
with st.sidebar:
    st.subheader("⚙️ Configuration")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your OpenRouter API key"
    )
    st.session_state.api_key = api_key_input
    
    st.divider()
    
    # Model selection
    st.subheader("Model Settings")
    selected_model = st.selectbox(
        "Model",
        options=list(OPENROUTER_IMAGE_MODELS.keys()),
        index=list(OPENROUTER_IMAGE_MODELS.keys()).index(default_model),
        help="Choose the image generation model"
    )
    
    # Aspect ratio
    use_auto_aspect = st.checkbox(
        "Auto-detect aspect ratio from reference image",
        value=True,
        help="Automatically use the aspect ratio of the first reference image"
    )
    
    aspect_ratio_display = st.selectbox(
        "Aspect Ratio",
        options=list(ASPECT_RATIOS.keys()),
        index=0,
        disabled=use_auto_aspect,
        help="Target aspect ratio for generated image"
    )
    aspect_ratio = ASPECT_RATIOS[aspect_ratio_display]
    
    # Seed
    st.subheader("Generation Parameters")
    use_random_seed = st.checkbox("Use random seed", value=True)
    if use_random_seed:
        seed = None
    else:
        seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=1000000,
            value=12345,
            help="Fixed seed for reproducible results"
        )
    
    # Image processing settings
    st.subheader("Reference Image Settings")
    max_image_size = st.slider(
        "Max reference image size (px)",
        min_value=256,
        max_value=2048,
        value=1024,
        step=128,
        help="Maximum dimension for reference images before encoding"
    )
    
    # Save settings
    st.divider()
    auto_save = st.checkbox(
        "Auto-save generated images",
        value=True,
        help="Automatically save images to outputs folder"
    )
    
    # Stealth Mode
    st.divider()
    stealth_mode = st.checkbox(
        "🕶️ Stealth Mode",
        value=False,
        help="Hide all image thumbnails for privacy"
    )

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
        if st.session_state.flattened_json_prompts:
            # Display JSON prompts
            json_categories = sorted(list(set([p['category'] for p in st.session_state.flattened_json_prompts])))
            
            selected_category = st.selectbox(
                "Select Prompt Template",
                options=json_categories,
                help="Choose a prompt template from JSON",
                key="json_category_select"
            )
            
            # Filter prompts by selected category
            filtered_prompts = [p for p in st.session_state.flattened_json_prompts if p['category'] == selected_category]
            
            if filtered_prompts:
                selected_prompt_obj = filtered_prompts[0]
                prompt = st.text_area(
                    "Selected Prompt (editable)",
                    value=selected_prompt_obj['prompt'],
                    height=200,
                    key=f"json_prompt_text_{selected_category}",
                    help="You can edit the loaded prompt before generating"
                )
            else:
                prompt = st.text_area(
                    "Image Prompt",
                    height=200,
                    placeholder="Select a prompt from the dropdown above...",
                    help="Select a prompt template from the dropdown",
                    key="empty_json_prompt"
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
        if st.session_state.flattened_prompts:
            # Group prompts by section
            sections = {}
            for item in st.session_state.flattened_prompts:
                section = item['section']
                if section not in sections:
                    sections[section] = []
                sections[section].append(item)
            
            # Create a hierarchical selector
            col_section, col_category = st.columns([1, 2])
            
            with col_section:
                section_options = list(sections.keys())
                selected_section = st.selectbox(
                    "Section",
                    options=section_options,
                    help="Choose between Create or Edit prompts"
                )
            
            # Get categories for selected section
            section_prompts = sections[selected_section]
            categories = sorted(list(set([p['category'] for p in section_prompts])))
            
            with col_category:
                selected_category = st.selectbox(
                    "Category",
                    options=categories,
                    help="Choose a category of prompts"
                )
            
            # Filter prompts by selected category
            filtered_prompts = [p for p in section_prompts if p['category'] == selected_category]
            
            # Create options for selectbox with preview
            prompt_options = ["Select a prompt..."] + [
                f"{p['prompt'][:80]}..." if len(p['prompt']) > 80 else p['prompt']
                for p in filtered_prompts
            ]
            
            selected_prompt_idx = st.selectbox(
                "Select Prompt",
                options=range(len(prompt_options)),
                format_func=lambda x: prompt_options[x],
                help="Choose a specific prompt"
            )
            
            if selected_prompt_idx > 0:
                selected_prompt_obj = filtered_prompts[selected_prompt_idx - 1]
                prompt = st.text_area(
                    "Selected Prompt (editable)",
                    value=selected_prompt_obj['prompt'],
                    height=150,
                    key="yaml_prompt_text",
                    help="You can edit the loaded prompt before generating"
                )
            else:
                prompt = st.text_area(
                    "Image Prompt",
                    height=150,
                    placeholder="Select a prompt from the dropdowns above...",
                    help="Select section, category and prompt from the dropdowns",
                    key="empty_yaml_prompt"
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
    
    # Reference images upload
    st.subheader("Reference Images (Optional)")
    uploaded_files = st.file_uploader(
        "Upload reference images",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more reference images to guide the generation"
    )
    
    # Display uploaded images
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} image(s) uploaded**")
        ref_cols = st.columns(min(len(uploaded_files), 3))
        reference_images = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file).convert("RGB")
            reference_images.append(img)
            
            with ref_cols[idx % 3]:
                if not stealth_mode:
                    st.image(img, caption=f"Ref {idx+1}", width=150, use_container_width=False)
                st.caption(f"Size: {img.size[0]}×{img.size[1]}")
    else:
        reference_images = None
    
    # Generate button
    generate_btn = st.button(
        "🎨 Generate Image",
        type="primary",
        use_container_width=True,
        disabled=not (prompt and st.session_state.api_key)
    )

with col2:
    st.subheader("🖼️ Generated Image")
    
    if generate_btn:
        if not st.session_state.api_key:
            st.error("❌ Please enter your OpenRouter API key in the sidebar")
        elif not prompt:
            st.error("❌ Please enter a prompt")
        else:
            with st.spinner("Generating image..."):
                try:
                    model_full_name = OPENROUTER_IMAGE_MODELS[selected_model]
                    
                    generated_image, used_aspect_ratio = generate_image(
                        prompt=prompt,
                        api_key=st.session_state.api_key,
                        model_name=model_full_name,
                        aspect_ratio=aspect_ratio,
                        seed=seed,
                        reference_images=reference_images,
                        use_image_aspect_ratio=use_auto_aspect,
                        max_image_size=max_image_size
                    )
                    
                    if generated_image:
                        st.success("✅ Image generated successfully!")
                        
                        # Display image
                        if not stealth_mode:
                            st.image(generated_image, use_container_width=True)
                        
                        # Image info
                        st.info(f"""
                        **Generation Details:**
                        - Model: `{selected_model}`
                        - Aspect Ratio: `{used_aspect_ratio}`
                        - Seed: `{seed if seed else 'Random'}`
                        - Reference Images: `{len(reference_images) if reference_images else 0}`
                        """)
                        
                        # Save image if auto-save is enabled
                        if auto_save:
                            saved_path = save_image_with_metadata(
                                generated_image,
                                prompt,
                                model_full_name,
                                seed if seed else random.randint(1, 1000000),
                                used_aspect_ratio
                            )
                            st.success(f"💾 Saved to: `{saved_path}`")

                            # autosave also comparison image
                            if reference_images:
                                comparison_img = create_comparison_image(
                                    generated_image, 
                                    reference_images,
                                    prompt=prompt,
                                    model=selected_model,
                                    seed=seed
                                )
                                if comparison_img:
                                    # save comparison as jpeg reducing quality
                                    comparison_path = saved_path.replace(".png", "_comparison.jpg")
                                    os.makedirs("outputs/comparisons", exist_ok=True)
                                    comparison_path = comparison_path.replace("outputs", "outputs/comparisons")
                                    print("Saving comparison image to:", comparison_path)
                                    comparison_img.save(comparison_path, format="JPEG", quality=75)
                                    st.success(f"💾 Comparison image saved to: `{comparison_path}`")
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # Download generated image
                            buf = BytesIO()
                            generated_image.save(buf, format="PNG")
                            st.download_button(
                                label="📥 Download Generated",
                                data=buf.getvalue(),
                                file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Download comparison if reference images exist
                            if reference_images:
                                comparison_img = create_comparison_image(
                                    generated_image, 
                                    reference_images,
                                    prompt=prompt,
                                    model=selected_model,
                                    seed=seed
                                )
                                if comparison_img:
                                    buf_comp = BytesIO()
                                    comparison_img.save(buf_comp, format="PNG")
                                    st.download_button(
                                        label="📥 Download Comparison",
                                        data=buf_comp.getvalue(),
                                        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                        
                        # Show comparison preview if reference images exist
                        # if reference_images and not stealth_mode:
                        with st.expander("👁️ View Comparison", expanded=False):
                            comparison_img = create_comparison_image(
                                generated_image, 
                                reference_images,
                                prompt=prompt,
                                model=selected_model,
                                seed=seed
                            )
                            if comparison_img:
                                st.image(comparison_img, caption="Reference(s) → Generated", use_container_width=True)
                    
                        # Add to history
                        st.session_state.generated_images.insert(0, {
                            'image': generated_image,
                            'prompt': prompt,
                            'model': selected_model,
                            'seed': seed,
                            'aspect_ratio': used_aspect_ratio,
                            'reference_images': reference_images,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                    else:
                        st.error("❌ Failed to generate image. No image data returned.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)


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
        qpg_task = st.selectbox(
            "Task",
            options=["GENERATE_PROMPT", "GENERATE_DETAILED_PROMPT", "GENERATE_JSON_PROMPT"],
            format_func=lambda x: {
                "GENERATE_PROMPT": "📝 Basic Prompt",
                "GENERATE_DETAILED_PROMPT": "📝 Detailed Prompt",
                "GENERATE_JSON_PROMPT": "📝 JSON Prompt"
            }.get(x, x),
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
        
        qpg_generate = st.button("🚀 Generate Prompt", type="primary", use_container_width=True, key="qpg_gen_btn")
    
    with qpg_col2:
        st.subheader("Generated Prompt")
        
        if qpg_generate and (qpg_draft or qpg_image):
            try:
                from promptgen_page import TaggerGPT, DEFAULT_SYSTEM_IMAGE_PROMPT, optimize_image
                
                with st.spinner(f"Generating with {qpg_model_key}..."):
                    tagger = TaggerGPT(qpg_model)
                    
                    # Build instruction
                    instruction = INSTUCTIONS[qpg_task]
                    
                    if qpg_draft:
                        instruction = f"{instruction}\n\nContext/Reference text: {qpg_draft}"
                    
                    # Process image if provided
                    processed_img = None
                    if qpg_image:
                        img = Image.open(qpg_image).convert("RGB")
                        processed_img = optimize_image(img, target_size=1120)
                    
                    # Generate
                    result = tagger.chat_completion_prompt(
                        DEFAULT_SYSTEM_IMAGE_PROMPT,
                        instruction,
                        image=processed_img
                    )
                    
                    st.success("✅ Prompt generated!")
                    
                    # Save to session state for immediate access
                    st.session_state['last_generated_prompt'] = result
                    
                    # Save to history
                    prompt_item = {
                        'result': result,
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
            result = st.session_state['last_generated_prompt']
            
            # Display result in text area (editable)
            st.text_area("Generated Result", value=result, height=200, key="qpg_result")

            if st.button("📋 Copy", key="copy_generated_result", use_container_width=True):
                try:
                    import pyperclip
                    pyperclip.copy(st.session_state['last_generated_prompt'])
                    st.success("✅ Copied!")
                except Exception as e:
                    # Display result in a code block with built-in copy button
                    st.code(result, language=None)
                    st.info("⚠️ Pyperclip not available. Use the code box copy button above.")
            
            # Download button
            st.download_button(
                "💾 Download Prompt",
                data=result,
                file_name="generated_prompt.txt",
                mime="text/plain",
                use_container_width=True,
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
        if st.button("🗑️ Clear Prompt History", use_container_width=True):
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
                if st.button("📋 Copy", key=f"copy_hist_{selected_hist_idx}", use_container_width=True):
                    try:
                        import pyperclip
                        pyperclip.copy(hist_item['result'])
                        st.success("✅ Copied!")
                    except:
                        st.info("Use code box")
            
            # Code block for easy copying
            st.code(hist_item['result'], language=None)



# Generation History

if st.session_state.generated_images:
    st.divider()
    st.subheader("📜 Generation History")
    
    for idx, item in enumerate(st.session_state.generated_images[:5]):
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
                # st.write(f"**Aspect Ratio:** {item.get('aspect_ratio', 'N/A')}")
                st.write(f"**Reference Images:** {len(item.get('reference_images', [])) if item.get('reference_images') else 0}")
                
                # Download buttons
                dl_cols = st.columns(2)
                
                with dl_cols[0]:
                    # Download from history
                    buf = BytesIO()
                    # in the image metadata must be saved the prompt info and model and seed
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("Prompt", item['prompt'])
                    metadata.add_text("Model", item['model'])
                    metadata.add_text("Seed", str(item.get('seed', 'N/A')))
                    # metadata.add_text("Aspect_Ratio", item.get('aspect_ratio', 'N/A'))
                    metadata.add_text("Timestamp", item['timestamp'])

                    item['image'].save(buf, format="PNG", pnginfo=metadata)
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name=f"history_{idx}.png",
                        mime="image/png",
                        key=f"download_history_{idx}"
                    )
                
                with dl_cols[1]:
                    # Download comparison if reference images exist
                    if item.get('reference_images'):
                        comparison_img = create_comparison_image(
                            item['image'],
                            item['reference_images'],
                            prompt=item['prompt'],
                            model=item['model'],
                            seed=item.get('seed')
                        )
                        if comparison_img:
                            buf_comp = BytesIO()
                            comparison_img.save(buf_comp, format="PNG")
                            st.download_button(
                                label="📥 Download Comparison",
                                data=buf_comp.getvalue(),
                                file_name=f"comparison_history_{idx}.png",
                                mime="image/png",
                                key=f"download_comparison_history_{idx}"
                            )


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with Streamlit • Powered by OpenRouter API</p>
</div>
""", unsafe_allow_html=True)
