import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image, ImageOps
import json
import os

# Instructions and settings
INSTUCTIONS = {
    "GENERATE_TAGS": "Your task is to generate general descriptive tags for the image. Provide a comma separated list of relevant keywords that capture the main elements, themes, and subjects present in the image. Answer only with the tags, without any additional explanation or description.",
    "GENERATE_DETAILED_TAGS": "Your task is to generate a detailed list of descriptive tags for the image. Provide a comprehensive comma separated list of relevant keywords that capture the main elements, themes, subjects, colors, and notable features present in the image. Answer only with the detailed tags, without any additional explanation or description.",
    "GENERATE_PROMPT": "Your task is to write the textual prompt from this image/photo. the textual prompt must be suitable for image generation models. Answer only with the prompt, without any additional explanation or description.",
    "GENERATE_DETAILED_PROMPT": "Your task is to write a highly detailed textual prompt from this image/photo. the textual prompt must be suitable for image generation models. Include specific details about the scene, subjects, colors, lighting, and any notable features. Answer only with the detailed prompt, without any additional explanation or description.",
    "DETAILED_CAPTION": "Your task is to provide a detailed caption for the image, describing its content, context, and any notable features in a clear and informative manner.",
    "MORE_DETAILED_CAPTION": "Your task is to provide an even more detailed caption for the image, elaborating on its content, context, and notable features with greater depth and specificity.",
}

if os.path.exists("tagger_focus.json"):
    with open("tagger_focus.json", 'r') as f:
        FOCUS = json.load(f)
else:        
    FOCUS = {
        "cuteness": "Focus on the cute and adorable elements of the image, detailing aspects that contribute to its charm and endearing qualities.",
        "artistic": "Focus on the artistic style and composition of the image, describing elements that highlight creativity, color usage, and visual aesthetics.",
        "fantasy": "Focus on the fantasy elements of the image, detailing aspects that contribute to its imaginative and otherworldly qualities.",
    }

client_dict = {
    "openai": "",
    "grok": "https://api.x.ai/v1",
    "claude": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1/",
    "deepseek": "https://api.deepseek.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

GROQ_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "llama-maverik-17b": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-scout-17b": "meta-llama/llama-4-scout-17b-16e-instruct",
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
}

OPENROUTER_MODELS = {
    "grok-4": "x-ai/grok-4",
    "grok-4-fast": "x-ai/grok-4-fast",
    "sonar-pro-search": "perplexity/sonar-pro-search",
    "bert-nebulon-alpha": "openrouter/bert-nebulon-alpha",  
}

XAI_MODELS = {
    "grok-3": "grok-3",
    "grok-4": "grok-4-0709"
}

DEFAULT_SYSTEM_IMAGE_PROMPT = "You are a helpful assistant that analyzes images and performs the requested tasks."

def load_api_keys(file_path="api_keys.json"):
    """Load API keys from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

api_dict = load_api_keys()

def get_client(client_name):
    """Initialize OpenAI client for specific provider"""
    api_key = api_dict.get(client_name, "")
    base_url = client_dict.get(client_name, "")
    return OpenAI(api_key=api_key, base_url=base_url)

def select_client_based_on_model(model_name):
    """Select appropriate client based on model name"""
    if model_name in GROQ_MODELS.values():
        return get_client("groq")
    elif model_name in XAI_MODELS.values():
        return get_client("grok")
    elif model_name in OPENROUTER_MODELS.values():
        return get_client("openrouter")
    else:
        return get_client("openai")

def resize_image(image, max_size=512, maintain_aspect=True):
    """Resize image conservatively, maintaining aspect ratio"""
    if maintain_aspect:
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        if width > max_size or height > max_size:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.copy()
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)

def optimize_image(image, target_size=1120):
    """Optimize image for vision models by resizing and padding to square"""
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2)
    )
    
    img_padded = ImageOps.expand(img_resized, padding, fill=(0, 0, 0))
    return img_padded

class TaggerGPT:
    """Streamlit-compatible TaggerGPT class"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = select_client_based_on_model(model_name)
        self.MAX_TOKENS = 8192
        self.TEMPERATURE = 0.7
        self.TOP_P = 0.9
        self.FREQUENCY_PENALTY = 0.0
        self.PRESENCE_PENALTY = 0.0
        self.SEED = 42

    def chat_completion_prompt(self, system_prompt, user_prompt, image=None):
        """Send a chat completion request with optional image support."""
        messages = [{"role": "system", "content": system_prompt}]

        if image:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        else:
            user_message = {"role": "user", "content": user_prompt}
        
        messages.append(user_message)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            top_p=self.TOP_P,
            # frequency_penalty=self.FREQUENCY_PENALTY,
            # presence_penalty=self.PRESENCE_PENALTY,
            seed=self.SEED,
        )
        message_content = response.choices[0].message.content
        return message_content


def show_tagger_page():
    """Main function to display the Prompt Generator page"""
    
    st.title("✨ Prompt Generator")
    st.markdown("Generate prompts from text, images, or both together using AI models")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Generator Settings")
        
        # Model selection
        st.subheader("Model Selection")
        
        model_category = st.selectbox(
            "Model Provider",
            ["OpenRouter", "Groq", "X.AI (Grok)"],
            help="Choose the AI provider"
        )
        
        if model_category == "OpenRouter":
            model_options = OPENROUTER_MODELS
        elif model_category == "Groq":
            model_options = GROQ_MODELS
        else:  # X.AI
            model_options = XAI_MODELS
        
        selected_model_key = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            help="Choose the vision model"
        )
        selected_model = model_options[selected_model_key]
        
        st.divider()
        
        # Task selection
        st.subheader("Task Selection")
        task_names = list(INSTUCTIONS.keys())
        task_labels = {
            "GENERATE_TAGS": "🏷️ General Tags",
            "GENERATE_DETAILED_TAGS": "🏷️ Detailed Tags",
            "GENERATE_PROMPT": "📝 Image Prompt",
            "GENERATE_DETAILED_PROMPT": "📝 Detailed Prompt",
            "DETAILED_CAPTION": "📋 Caption",
            "MORE_DETAILED_CAPTION": "📋 Detailed Caption"
        }
        
        selected_task = st.selectbox(
            "Task Type",
            options=task_names,
            format_func=lambda x: task_labels.get(x, x),
            help="Choose what type of output you want"
        )
        
        st.divider()
        
        # Focus/Style options
        st.subheader("Focus Options")
        use_focus = st.checkbox("Add focus style", value=False)
        
        if use_focus:
            focus_options = list(FOCUS.keys())
            selected_focus = st.selectbox(
                "Focus Style",
                options=focus_options,
                help="Add specific focus to the analysis"
            )
        else:
            selected_focus = None
        
        st.divider()
        
        # Image optimization settings
        st.subheader("Image Processing")
        optimize_images = st.checkbox(
            "Optimize images for vision models",
            value=True,
            help="Resize and pad images to optimal dimensions (1120x1120)"
        )
        
        if optimize_images:
            target_size = st.slider(
                "Target size (px)",
                min_value=560,
                max_value=2240,
                value=1120,
                step=560,
                help="560 for 1 tile, 1120 for 4 tiles"
            )
        else:
            target_size = st.slider(
                "Max dimension (px)",
                min_value=256,
                max_value=2048,
                value=1024,
                step=128
            )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Draft text input
        draft_text = st.text_area(
            "Draft Text (Optional)",
            height=150,
            placeholder="e.g., A girl sitting on a bench in a park...",
            help="Enter your draft text (optional). Works with or without images."
        )
        
        st.divider()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Images (Optional)",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            help="Upload images to analyze (optional). Works with or without text."
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} image(s) uploaded**")
            
            # Display thumbnails
            thumbnail_cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                with thumbnail_cols[idx % 3]:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                    st.caption(f"Size: {img.size[0]}×{img.size[1]}")
        
        st.divider()
        
        # Additional instructions
        st.subheader("Additional Instructions (Optional)")
        additional_prompt = st.text_area(
            "Add custom instructions",
            height=100,
            placeholder="e.g., Focus on specific details, add more style...",
            help="Add any additional context or instructions for the model"
        )
        
        # Generate button
        has_draft_text = draft_text and draft_text.strip()
        has_images = uploaded_files is not None and len(uploaded_files) > 0
        can_generate = has_draft_text or has_images
        
        process_btn = st.button(
            "🚀 Generate",
            type="primary",
            use_container_width=True,
            disabled=not can_generate
        )
    
    with col2:
        st.header("📄 Generated Results")
        
        if process_btn and can_generate:
            # Initialize tagger
            try:
                with st.spinner(f"Initializing {selected_model_key} model..."):
                    tagger = TaggerGPT(selected_model)
                
                results = []
                
                # Case 1: Only text provided (no images)
                if has_draft_text and not has_images:
                    st.subheader("Generated Prompt")
                    
                    # Build prompt for text enhancement
                    text_instruction = f"Enhance and refine the following draft text into a polished prompt suitable for image generation. Maintain the core idea while adding appropriate details, style, and clarity."
                    
                    if use_focus and selected_focus:
                        focus_text = FOCUS[selected_focus]
                        text_instruction = f"{text_instruction}\n\n{focus_text}"
                    
                    if additional_prompt:
                        text_instruction = f"{text_instruction}\n\n{additional_prompt}"
                    
                    text_instruction = f"{text_instruction}\n\nDraft text: {draft_text}"
                    
                    # Generate response
                    with st.spinner("Generating enhanced prompt..."):
                        try:
                            result = tagger.chat_completion_prompt(
                                DEFAULT_SYSTEM_IMAGE_PROMPT,
                                text_instruction,
                                image=None
                            )
                            
                            st.success("✅ Prompt generated successfully")
                            
                            # Display result
                            st.markdown("**Enhanced Prompt:**")
                            st.info(result)
                            
                            # Copy button
                            st.code(result, language=None)
                            
                            results.append({
                                'filename': 'text_prompt',
                                'result': result,
                                'task': 'Text Enhancement'
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error generating prompt: {str(e)}")
                
                # Case 2: Images provided (with or without text)
                elif has_images:
                    # Build the base prompt
                    base_instruction = INSTUCTIONS[selected_task]
                    
                    if use_focus and selected_focus:
                        focus_text = FOCUS[selected_focus]
                        user_prompt = f"{base_instruction}\n\n{focus_text}"
                    else:
                        user_prompt = base_instruction
                    
                    # Add draft text if provided
                    if has_draft_text:
                        user_prompt = f"{user_prompt}\n\nContext/Reference text: {draft_text}"
                    
                    if additional_prompt:
                        user_prompt = f"{user_prompt}\n\n{additional_prompt}"
                    
                    # Process each image
                    for idx, uploaded_file in enumerate(uploaded_files):
                        st.subheader(f"Image {idx+1}: {uploaded_file.name}")
                        
                        # Load and process image
                        img = Image.open(uploaded_file).convert("RGB")
                        
                        # Show preview
                        with st.expander("🖼️ View Image", expanded=False):
                            st.image(img, use_container_width=True)
                        
                        # Optimize or resize image
                        if optimize_images:
                            with st.spinner(f"Optimizing image {idx+1}..."):
                                processed_img = optimize_image(img, target_size=target_size)
                        else:
                            processed_img = resize_image(img, max_size=target_size)
                        
                        # Generate response
                        with st.spinner(f"Analyzing image {idx+1}..."):
                            try:
                                result = tagger.chat_completion_prompt(
                                    DEFAULT_SYSTEM_IMAGE_PROMPT,
                                    user_prompt,
                                    image=processed_img
                                )
                                
                                st.success(f"✅ Generated for image {idx+1}")
                                
                                # Display result
                                st.markdown("**Result:**")
                                st.info(result)
                                
                                # Copy button
                                st.code(result, language=None)
                                
                                results.append({
                                    'filename': uploaded_file.name,
                                    'result': result,
                                    'task': selected_task
                                })
                                
                            except Exception as e:
                                st.error(f"❌ Error processing image {idx+1}: {str(e)}")
                        
                        st.divider()
                
                # Summary and export
                if results:
                    st.subheader("📊 Batch Summary")
                    st.write(f"Successfully processed **{len(results)}** images")
                    
                    # Export all results
                    export_text = "\n\n" + "="*50 + "\n\n"
                    for r in results:
                        export_text += f"FILE: {r['filename']}\n"
                        export_text += f"TASK: {r['task']}\n\n"
                        export_text += f"{r['result']}\n"
                        export_text += "\n" + "="*50 + "\n\n"
                    
                    st.download_button(
                        label="📥 Download All Results",
                        data=export_text,
                        file_name="tagger_results.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
        
        else:
            st.info("👈 Enter draft text and/or upload images to get started")
    
    # Info section
    st.divider()
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        ### How It Works:
        
        - **Text Only**: Enter draft text to enhance it into a polished prompt
        - **Images Only**: Upload images to generate prompts, tags, or captions
        - **Text + Images**: Combine both for context-aware image analysis
        
        ### Available Tasks:
        
        - **General Tags**: Generate comma-separated keywords capturing main elements
        - **Detailed Tags**: Comprehensive list including colors, themes, and features
        - **Image Prompt**: Generate a text prompt suitable for image generation models
        - **Detailed Prompt**: Highly detailed prompt with specific scene descriptions
        - **Caption**: Detailed caption describing content and context
        - **Detailed Caption**: Even more detailed caption with greater depth
        
        ### Focus Options:
        
        - **Cuteness**: Focus on cute and adorable elements
        - **Artistic**: Emphasize artistic style and composition
        - **Fantasy**: Highlight imaginative and otherworldly qualities
        - (You can add custom focus styles in `tagger_focus.json`)"
        
        ### Image Optimization:
        
        Vision models work best with specific image sizes:
        - **560x560**: Single tile view (minimal detail)
        - **1120x1120**: 4-tile view (recommended, good detail)
        - **2240x2240**: 16-tile view (maximum detail, slower)
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Prompt Generator • Powered by multiple AI providers</p>
    </div>
    """, unsafe_allow_html=True)
