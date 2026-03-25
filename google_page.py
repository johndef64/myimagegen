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
default_model = "gemini-3.1-flash-image-preview"

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

OUTPUT_RESOLUTIONS = {
    "0.5K (512px)":  "512",
    "1K (1024px)":   "1K",
    "2K (2048px)":   "2K",
    "4K (4096px)":   "4K",
}
DEFAULT_RESOLUTION = "1K (1024px)"

SAFETY_ACTION = "BLOCK_NONE"
SAFETY_ACTION = "OFF"
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=SAFETY_ACTION),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=SAFETY_ACTION),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=SAFETY_ACTION),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=SAFETY_ACTION),
    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold=SAFETY_ACTION),
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
                          max_image_size=1024, output_resolution="1024"):
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
        return _generate_imagen(client, model_id, prompt, aspect_ratio, seed, output_resolution)

    # Gemini models - use generate_content with image modality
    contents = []

    # Add reference images if provided and model supports it.
    # ORDER IS PRESERVED: images are added in the same order as displayed in the UI
    # so prompts can safely reference "the first image", "the second image", etc.
    if reference_images and model_key in MODELS_WITH_IMG2IMG:
        for ref_img in reference_images:
            resized = resize_image(ref_img, max_image_size)
            img_bytes = pil_to_bytes(resized, format="JPEG")
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

    # Text prompt is always added AFTER all images
    contents.append(prompt)

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=SAFETY_SETTINGS,
            seed=seed,
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=output_resolution,
            ),
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


def _generate_imagen(client, model_id, prompt, aspect_ratio, seed, output_resolution="1024"):
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
                image_size=output_resolution,
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


# ─────────────────────────────────────────────────────────────────────────────
# Batch Generator helpers
# ─────────────────────────────────────────────────────────────────────────────

# Cost per generated image at each resolution (Batch API pricing, USD)
BATCH_IMAGE_COSTS = {
    "512": 0.022,   # 0.5K
    "1K":  0.034,
    "2K":  0.050,
    "4K":  0.076,
}

# Models that work with Batch image generation (Gemini multimodal, not Imagen)
BATCH_COMPATIBLE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
]


def _build_batch_jsonl(queue):
    """Build a list of JSONL request dicts from the batch queue.

    Each queue entry may request N images; we expand that into N individual
    batch requests (one image per request) with unique keys.
    Reference images (PIL) are base64-encoded inline.
    """
    import base64 as _b64

    requests_data = []
    req_idx = 1
    for job in queue:
        ref_parts = []
        for ref_img in job.get('ref_images') or []:
            buf = BytesIO()
            resized = resize_image(ref_img, 1024)
            resized.save(buf, format="JPEG")
            b64 = _b64.b64encode(buf.getvalue()).decode('utf-8')
            ref_parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": b64}
            })

        for _ in range(job['num_images']):
            parts = ref_parts + [{"text": job['prompt']}]
            requests_data.append({
                "key": f"img_{req_idx}",
                "request": {
                    "contents": [{"parts": parts}],
                    "generation_config": {
                        "response_modalities": ["TEXT", "IMAGE"],
                        "safety_settings":SAFETY_SETTINGS,
                        "image_config": {
                            "aspect_ratio": job['aspect_ratio'],
                            "image_size": job['resolution'],
                        }
                    }
                }
            })
            req_idx += 1
    return requests_data



BATCH_LOG_FILE = "batch_jobs_log.json"


def _batch_log_load():
    """Load persisted batch job records from disk (strips non-serialisable PIL images)."""
    if not os.path.exists(BATCH_LOG_FILE):
        return []
    try:
        with open(BATCH_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _batch_log_save(jobs):
    """Persist batch job records to disk.

    PIL Image objects and raw image bytes are not JSON-serialisable, so we
    strip them before writing (result_images bytes are saved separately as
    PNG files; PIL ref_images are never persisted).
    """
    serialisable = []
    for job in jobs:
        j = {k: v for k, v in job.items() if k not in ('result_images',)}
        # Strip ref_images (PIL objects) from nested request list too
        requests_clean = []
        for req in j.get('requests', []):
            requests_clean.append({k: v for k, v in req.items() if k != 'ref_images'})
        j['requests'] = requests_clean
        # Persist paths to already-saved images instead of raw bytes
        saved_paths = job.get('saved_image_paths', [])
        j['saved_image_paths'] = saved_paths
        serialisable.append(j)
    try:
        with open(BATCH_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _batch_save_images_to_disk(job_record):
    """Save fetched result images to outputs/batch/ with descriptive filenames and PNG metadata."""
    result_images = job_record.get('result_images', [])
    if not result_images:
        return

    safe_ts = job_record.get('submitted_at', 'batch').replace(' ', '_').replace(':', '-')
    model_short = job_record.get('model', 'batch').split('/')[-1]
    folder = os.path.join("outputs", "batch", safe_ts)
    os.makedirs(folder, exist_ok=True)

    # Build a key→prompt lookup from the requests list so we can embed prompt in metadata
    # Each request entry covers num_images images; expand into per-key mapping
    key_to_prompt = {}
    req_idx = 1
    for req in job_record.get('requests', []):
        for _ in range(req.get('num_images', 1)):
            key_to_prompt[f"img_{req_idx}"] = req.get('prompt', '')
            req_idx += 1

    paths = []
    for img_i, img_data in enumerate(result_images):
        key = img_data.get('key', f'img_{img_i + 1}')
        prompt = key_to_prompt.get(key, '')
        prompt_slug = prompt[:40].replace(' ', '_').replace('\n', '_')
        prompt_slug = ''.join(c for c in prompt_slug if c.isalnum() or c in ('_', '-'))

        fname = os.path.join(
            folder,
            f"{key}_{model_short}_{prompt_slug}.png" if prompt_slug else f"{key}_{model_short}.png"
        )

        try:
            pil_img = Image.open(BytesIO(img_data['data']))
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Prompt", prompt)
            metadata.add_text("Model", job_record.get('model', ''))
            metadata.add_text("BatchJob", job_record.get('job_name', ''))
            metadata.add_text("Key", key)
            metadata.add_text("SubmittedAt", job_record.get('submitted_at', ''))
            metadata.add_text("Provider", "Google AI Studio Batch API")
            pil_img.save(fname, format="PNG", pnginfo=metadata)
            paths.append(fname)
        except Exception:
            paths.append(None)

    job_record['saved_image_paths'] = paths


def submit_google_batch_jobs(queue, api_key):
    """Upload JSONL and create a Google Batch job. Returns a job record dict."""
    import tempfile
    client = get_google_client(api_key)

    # Build JSONL
    requests_data = _build_batch_jsonl(queue)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for req in requests_data:
            f.write(json.dumps(req) + '\n')
        tmp_path = f.name

    # Upload file to Files API (mime_type must be explicit — SDK can't infer .jsonl)
    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(
            display_name='batch-image-gen-input',
            mime_type='application/jsonl',
        )
    )
    os.unlink(tmp_path)

    # Determine model from first job in queue (all jobs in same batch must use same model)
    model_id = queue[0]['model']

    batch_job = client.batches.create(
        model=model_id,
        src=uploaded.name,
        config={'display_name': f'image-batch-{datetime.now().strftime("%Y%m%d-%H%M%S")}'}
    )

    total_images = sum(j['num_images'] for j in queue)
    total_cost = sum(j['cost'] for j in queue)

    job_record = {
        'job_name': batch_job.name,
        'state': batch_job.state.name,
        'submitted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_images': total_images,
        'total_cost': total_cost,
        'model': model_id,
        'requests': [
            {
                'prompt': j['prompt'],
                'num_images': j['num_images'],
                'model': j['model'],
                'resolution': j['resolution'],
                'resolution_display': j['resolution_display'],
            }
            for j in queue
        ],
        'result_images': [],
        'saved_image_paths': [],
    }

    # Persist to disk immediately so the job survives app restarts
    all_jobs = _batch_log_load()
    all_jobs.insert(0, job_record)
    _batch_log_save(all_jobs)

    return job_record


def refresh_google_batch_job(job_record, api_key):
    """Poll the API and update the state of a job record in-place."""
    client = get_google_client(api_key)
    try:
        batch_job = client.batches.get(name=job_record['job_name'])
        job_record['state'] = batch_job.state.name
        if batch_job.state.name == 'JOB_STATE_FAILED':
            job_record['error'] = str(getattr(batch_job, 'error', ''))
    except Exception as e:
        job_record['state'] = 'ERROR'
        job_record['error'] = str(e)
    # Persist updated state
    _batch_log_sync_one(job_record)
    return job_record


def fetch_google_batch_results(job_record, api_key):
    """Download result file, extract image bytes, save to disk. Returns updated job_record."""
    import base64 as _b64
    client = get_google_client(api_key)
    try:
        batch_job = client.batches.get(name=job_record['job_name'])
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            return job_record

        result_file_name = batch_job.dest.file_name
        file_bytes = client.files.download(file=result_file_name)
        content = file_bytes.decode('utf-8')

        result_images = []
        for line in content.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            key = parsed.get('key', '')
            candidates = parsed.get('response', {}).get('candidates', [])
            for candidate in candidates:
                parts = candidate.get('content', {}).get('parts', [])
                for part in parts:
                    inline = part.get('inlineData') or part.get('inline_data')
                    if inline:
                        img_data = _b64.b64decode(inline['data'])
                        result_images.append({
                            'key': key,
                            'data': img_data,
                            'mime_type': inline.get('mimeType', 'image/png'),
                        })
        job_record['result_images'] = result_images
        job_record['state'] = 'JOB_STATE_SUCCEEDED'

        # Save images to disk and persist paths in the log
        _batch_save_images_to_disk(job_record)
        _batch_log_sync_one(job_record)

    except Exception as e:
        job_record['fetch_error'] = str(e)
        _batch_log_sync_one(job_record)
    return job_record


def _batch_log_sync_one(job_record):
    """Update a single job record in the persisted log by job_name."""
    all_jobs = _batch_log_load()
    target = job_record.get('job_name')
    updated = False
    for i, j in enumerate(all_jobs):
        if j.get('job_name') == target:
            # Merge state/error/saved_image_paths without overwriting serialisable fields
            j['state'] = job_record.get('state', j.get('state'))
            if 'error' in job_record:
                j['error'] = job_record['error']
            if 'fetch_error' in job_record:
                j['fetch_error'] = job_record['fetch_error']
            j['saved_image_paths'] = job_record.get('saved_image_paths', j.get('saved_image_paths', []))
            all_jobs[i] = j
            updated = True
            break
    if not updated:
        all_jobs.insert(0, job_record)
    _batch_log_save(all_jobs)


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
    if 'google_batch_queue' not in st.session_state:
        st.session_state.google_batch_queue = []
    if 'google_batch_jobs' not in st.session_state:
        # Restore persisted jobs from disk; result_images are reloaded from saved_image_paths
        persisted = _batch_log_load()
        for jr in persisted:
            jr.setdefault('result_images', [])
            # Re-populate result_images from previously saved PNG files
            if not jr['result_images'] and jr.get('saved_image_paths'):
                for path in jr['saved_image_paths']:
                    if path and os.path.exists(path):
                        try:
                            with open(path, 'rb') as f:
                                jr['result_images'].append({
                                    'key': os.path.splitext(os.path.basename(path))[0],
                                    'data': f.read(),
                                    'mime_type': 'image/png',
                                })
                        except Exception:
                            pass
        st.session_state.google_batch_jobs = persisted

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

        # Output resolution
        output_resolution_display = st.selectbox(
            "Output Resolution",
            options=list(OUTPUT_RESOLUTIONS.keys()),
            index=list(OUTPUT_RESOLUTIONS.keys()).index(DEFAULT_RESOLUTION),
            help="Resolution of the generated image. 1K is the default.",
            key="google_output_resolution"
        )
        output_resolution = OUTPUT_RESOLUTIONS[output_resolution_display]

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

                section_keys = {"Create": "create_prompts", 
                                "Edit": "edit_prompts", 
                                "Qwen Edit": "qwen_edit",
                                "NanoBanana Edit": "nano_banana_edit"}
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
            # st.info(
            #     "The images are sent to the model **in this exact order** (left to right, top to bottom). "
            #     "Use 'the first image', 'the second image', etc. in your prompt to reference them.",
            #     icon="ℹ️"
            # )
            ref_cols = st.columns(min(len(uploaded_files), 3))
            reference_images = []

            for idx, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                reference_images.append(img)

                with ref_cols[idx % 3]:
                    if not stealth_mode:
                        st.image(img, caption=f"#{idx+1} — {uploaded_file.name}", width=150)
                    st.caption(f"Request position: **{idx+1}** | {img.size[0]}x{img.size[1]}")
        elif not uploaded_files:
            reference_images = None



    with col2:
        st.subheader("🖼️ Generated Image")
        # Generate button
        generate_btn = st.button(
            "🎨 Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not (prompt and st.session_state.google_api_key),
            key="google_generate_btn"
        )

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
                            max_image_size=max_image_size,
                            output_resolution=output_resolution
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
                            - Resolution: `{output_resolution_display}`
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
                                dl_meta = PngImagePlugin.PngInfo()
                                dl_meta.add_text("Prompt", prompt)
                                dl_meta.add_text("Model", selected_model)
                                dl_meta.add_text("Seed", str(seed if seed else ""))
                                dl_meta.add_text("Aspect_Ratio", used_aspect_ratio)
                                dl_meta.add_text("Provider", "Google AI Studio")
                                generated_image.save(buf, format="PNG", pnginfo=dl_meta)
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

    # ─── Batch Generator Section ──────────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Batch Image Generator")
    st.markdown(
        "Generate images in bulk via the **Google Batch API** — "
        "up to **50% cheaper** than standard calls, async delivery within 24h."
    )

    # ── Job Builder ──
    with st.expander("➕ Build Batch Queue", expanded=True):
        bq_col1, bq_col2 = st.columns([2, 1])

        with bq_col1:
            batch_prompt_input = st.text_area(
                "Image Prompt",
                height=120,
                placeholder="Describe the image(s) you want to generate...",
                key="google_batch_prompt_input"
            )

            batch_ref_files = st.file_uploader(
                "Reference Images (Optional)",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                accept_multiple_files=True,
                help="Upload reference images to include in each request of this job",
                key="google_batch_ref_upload"
            )
            batch_ref_images = []
            if batch_ref_files:
                ref_preview_cols = st.columns(min(len(batch_ref_files), 4))
                for ri, rf in enumerate(batch_ref_files):
                    img = Image.open(rf)
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    batch_ref_images.append(img)
                    with ref_preview_cols[ri % 4]:
                        if not stealth_mode:
                            st.image(img, caption=f"#{ri+1} {rf.name}", width=100)
                        else:
                            st.caption(f"Reference Image #{ri+1}")

        with bq_col2:
            batch_model_input = st.selectbox(
                "Model",
                options=BATCH_COMPATIBLE_MODELS,
                index=1,
                help="Only Gemini multimodal models support Batch image generation",
                key="google_batch_model_input"
            )

            batch_res_display = st.selectbox(
                "Resolution",
                options=list(OUTPUT_RESOLUTIONS.keys()),
                index=list(OUTPUT_RESOLUTIONS.keys()).index(DEFAULT_RESOLUTION),
                key="google_batch_res_input"
            )
            batch_res_val = OUTPUT_RESOLUTIONS[batch_res_display]

            batch_num_images = st.number_input(
                "Number of Images",
                min_value=1, max_value=500, value=1,
                help="Each image becomes one request in the batch",
                key="google_batch_num_images"
            )

            batch_use_auto_aspect = st.checkbox(
                "Auto-detect aspect ratio",
                value=True,
                help="Uses the aspect ratio of the first reference image uploaded above",
                key="google_batch_auto_aspect"
            )

            batch_aspect_display = st.selectbox(
                "Aspect Ratio",
                options=list(ASPECT_RATIOS.keys()),
                index=0,
                disabled=batch_use_auto_aspect,
                key="google_batch_aspect_input"
            )
            if batch_use_auto_aspect and batch_ref_images:
                batch_aspect_val = get_image_aspect_ratio(batch_ref_images[0])
            else:
                batch_aspect_val = ASPECT_RATIOS[batch_aspect_display]

        # Cost / time preview
        cost_per_img = BATCH_IMAGE_COSTS.get(batch_res_val, 0.034)
        job_cost = cost_per_img * batch_num_images
        st.info(
            f"**Cost estimate:** ${job_cost:.4f}  "
            f"({batch_num_images} image(s) × ${cost_per_img:.3f} @ {batch_res_display})  |  "
            f"**Delivery:** up to 24 h  |  "
            f"**Discount:** ~50% vs standard API"
        )

        if st.button("+ Add to Queue", type="secondary", use_container_width=False,
                     key="google_batch_add_job_btn"):
            if not batch_prompt_input.strip():
                st.error("Please enter a prompt before adding to queue.")
            else:
                entry = {
                    'id': len(st.session_state.google_batch_queue) + 1,
                    'prompt': batch_prompt_input.strip(),
                    'model': batch_model_input,
                    'resolution': batch_res_val,
                    'resolution_display': batch_res_display,
                    'num_images': int(batch_num_images),
                    'aspect_ratio': batch_aspect_val,
                    'cost': job_cost,
                    'ref_images': batch_ref_images,  # list of PIL images
                }
                st.session_state.google_batch_queue.append(entry)
                ref_note = f" + {len(batch_ref_images)} ref image(s)" if batch_ref_images else ""
                st.success(f"✅ Added job #{entry['id']} to queue ({batch_num_images} image(s){ref_note})")
                st.rerun()

    # ── Queue view ──
    if st.session_state.google_batch_queue:
        queue_total_images = sum(j['num_images'] for j in st.session_state.google_batch_queue)
        queue_total_cost = sum(j['cost'] for j in st.session_state.google_batch_queue)

        st.subheader(f"📋 Queue — {len(st.session_state.google_batch_queue)} job(s) | "
                     f"{queue_total_images} images | Est. ${queue_total_cost:.4f}")

        for idx, job in enumerate(st.session_state.google_batch_queue):
            jc1, jc2, jc3 = st.columns([5, 2, 1])
            with jc1:
                n_refs = len(job.get('ref_images') or [])
                ref_badge = f" | 🖼️ {n_refs} ref(s)" if n_refs else ""
                st.write(
                    f"**#{job['id']}** `{job['model']}` — "
                    f"{job['num_images']}x @ {job['resolution_display']} ({job['aspect_ratio']}){ref_badge}"
                )
                st.caption(f"> {job['prompt'][:120]}{'...' if len(job['prompt'])>120 else ''}")
            with jc2:
                st.caption(f"Est. cost: **${job['cost']:.4f}**")
            with jc3:
                if st.button("🗑️", key=f"google_batch_rm_{idx}_{job['id']}",
                             help="Remove from queue"):
                    st.session_state.google_batch_queue.pop(idx)
                    st.rerun()

        st.divider()
        bsub_col, bclr_col = st.columns([3, 1])

        with bclr_col:
            if st.button("🗑️ Clear Queue", use_container_width=True,
                         key="google_batch_clear_queue_btn"):
                st.session_state.google_batch_queue = []
                st.rerun()

        with bsub_col:
            # Warn if multiple models are mixed — batch API requires single model per job
            models_in_queue = list({j['model'] for j in st.session_state.google_batch_queue})
            if len(models_in_queue) > 1:
                st.warning(
                    f"⚠️ Queue contains multiple models ({', '.join(models_in_queue)}). "
                    "The Batch API requires a single model per job. "
                    "Only the first model will be used."
                )

            submit_disabled = not st.session_state.google_api_key
            if st.button(
                f"🚀 Submit Batch Job ({queue_total_images} images, est. ${queue_total_cost:.4f})",
                type="primary", use_container_width=True,
                disabled=submit_disabled,
                key="google_batch_submit_btn"
            ):
                if not st.session_state.google_api_key:
                    st.error("❌ Please enter your Google AI API key in the sidebar first.")
                else:
                    with st.spinner("Uploading requests and creating batch job..."):
                        try:
                            job_record = submit_google_batch_jobs(
                                st.session_state.google_batch_queue,
                                st.session_state.google_api_key
                            )
                            st.session_state.google_batch_jobs.insert(0, job_record)
                            st.session_state.google_batch_queue = []
                            st.success(
                                f"✅ Batch job created: `{job_record['job_name']}`  |  "
                                f"State: `{job_record['state']}`"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to submit batch job: {e}")
                            st.exception(e)

    # ── Jobs monitor ──
    st.divider()
    st.subheader("📊 Batch Jobs Monitor")

    # ── Retention notice ──────────────────────────────────────────────────────
    st.info(
        "⚠️ **Result file retention:** Google's Files API retains output files for ~**48 hours** "
        "after a batch job completes (same TTL as the Files API). "
        "Jobs themselves expire after **48 hours** if still pending/running. "
        "Download your images as soon as the job succeeds — "
        "after ~48 h the result file will be deleted by Google automatically.\n\n"
        "Jobs are logged locally in `batch_jobs_log.json` and restored on app restart, "
        "but raw image bytes are only available while the result file still exists on Google's servers."
    )

    mon_c1, mon_c2, mon_c3, mon_c4 = st.columns([1, 1, 1, 1])
    with mon_c1:
        if st.button("🔄 Refresh All", use_container_width=True,
                     key="google_batch_refresh_all"):
            if st.session_state.google_api_key:
                with st.spinner("Refreshing job statuses..."):
                    for jr in st.session_state.google_batch_jobs:
                        refresh_google_batch_job(jr, st.session_state.google_api_key)
                st.rerun()
            else:
                st.error("❌ API key required.")

    with mon_c2:
        if st.button("☁️ Import from Google", use_container_width=True,
                     key="google_batch_import_from_api",
                     help="Fetch recent batch jobs directly from Google API and merge into the log"):
            if not st.session_state.google_api_key:
                st.error("❌ API key required.")
            else:
                with st.spinner("Fetching batch jobs from Google API..."):
                    try:
                        _client = get_google_client(st.session_state.google_api_key)
                        remote_batches = _client.batches.list(config={'page_size': 100})
                        known_names = {jr['job_name'] for jr in st.session_state.google_batch_jobs}
                        imported = 0
                        for b in remote_batches.page:
                            if b.name not in known_names:
                                new_jr = {
                                    'job_name': b.name,
                                    'state': b.state.name,
                                    'submitted_at': b.create_time.strftime("%Y-%m-%d %H:%M:%S")
                                        if hasattr(b, 'create_time') and b.create_time else 'unknown',
                                    'total_images': 0,
                                    'total_cost': 0.0,
                                    'model': getattr(b, 'model', 'unknown'),
                                    'requests': [],
                                    'result_images': [],
                                    'saved_image_paths': [],
                                    'imported': True,
                                }
                                st.session_state.google_batch_jobs.insert(0, new_jr)
                                known_names.add(b.name)
                                imported += 1
                        # Persist merged list
                        _batch_log_save(st.session_state.google_batch_jobs)
                        st.success(f"✅ Imported {imported} new job(s) from Google API.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to import: {e}")

    with mon_c3:
        if st.button("🗑️ Clear Completed", use_container_width=True,
                     key="google_batch_clear_done"):
            done_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
            st.session_state.google_batch_jobs = [
                jr for jr in st.session_state.google_batch_jobs
                if jr.get('state') not in done_states
            ]
            _batch_log_save(st.session_state.google_batch_jobs)
            st.rerun()

    with mon_c4:
        if st.button("🗑️ Clear All", use_container_width=True,
                     key="google_batch_clear_all_jobs"):
            st.session_state.google_batch_jobs = []
            _batch_log_save([])
            st.rerun()

    if not st.session_state.google_batch_jobs:
        st.info("No batch jobs yet. Submit a batch above.")
    else:

        for job_idx, job_record in enumerate(st.session_state.google_batch_jobs):
            state = job_record.get('state', 'UNKNOWN')
            state_icon = {
                'JOB_STATE_SUCCEEDED': '✅',
                'JOB_STATE_FAILED': '❌',
                'JOB_STATE_CANCELLED': '⚠️',
                'JOB_STATE_RUNNING': '▶️',
                'JOB_STATE_PENDING': '⏳',
                'ERROR': '🔥',
            }.get(state, '⏳')

            expander_label = (
                f"{state_icon} {job_record['submitted_at']}  |  "
                f"{job_record['total_images']} image(s)  |  "
                f"Model: {job_record['model']}  |  "
                f"State: {state}"
            )
            with st.expander(expander_label, expanded=(state == 'JOB_STATE_SUCCEEDED')):
                info_c1, info_c2 = st.columns([2, 1])
                with info_c1:
                    st.write(f"**Job name:** `{job_record.get('job_name', 'N/A')}`")
                    if job_record.get('imported'):
                        st.caption("☁️ Recovered from Google API")
                    saved_paths = [p for p in job_record.get('saved_image_paths', []) if p]
                    if saved_paths:
                        st.caption(f"💾 {len(saved_paths)} image(s) saved locally in `outputs/batch/`")
                    if job_record.get('error'):
                        st.error(f"Error: {job_record['error']}")
                    if job_record.get('fetch_error'):
                        st.warning(f"Fetch error: {job_record['fetch_error']}")
                with info_c2:
                    st.metric("Images", job_record['total_images'])
                    st.metric("Est. cost", f"${job_record['total_cost']:.4f}")

                st.write("**Requests:**")
                for req in job_record.get('requests', []):
                    st.caption(
                        f"• {req['num_images']}× `{req['model']}` "
                        f"@ {req['resolution_display']} — "
                        f"{req['prompt'][:80]}{'...' if len(req['prompt'])>80 else ''}"
                    )

                jr_col1, jr_col2, jr_col3 = st.columns([1, 1, 2])

                with jr_col1:
                    if st.button("🔄 Refresh", key=f"google_batch_refresh_{job_idx}",
                                 use_container_width=True):
                        if st.session_state.google_api_key:
                            refresh_google_batch_job(job_record, st.session_state.google_api_key)
                            st.rerun()

                with jr_col2:
                    if state == 'JOB_STATE_SUCCEEDED' and not job_record.get('result_images'):
                        if st.button("📥 Fetch Images", key=f"google_batch_fetch_{job_idx}",
                                     use_container_width=True, type="primary"):
                            with st.spinner("Downloading results..."):
                                fetch_google_batch_results(job_record, st.session_state.google_api_key)
                            saved = [p for p in job_record.get('saved_image_paths', []) if p]
                            if saved:
                                safe_ts = job_record.get('submitted_at', '').replace(' ', '_').replace(':', '-')
                                st.success(
                                    f"💾 {len(saved)} image(s) auto-saved to "
                                    f"`outputs/batch/{safe_ts}/`"
                                )
                            st.rerun()

                # Display fetched images
                result_images = job_record.get('result_images', [])
                if result_images:
                    st.write(f"**Generated Images ({len(result_images)}):**")
                    img_per_row = 4
                    img_cols = st.columns(img_per_row)
                    for img_i, img_data in enumerate(result_images):

                        with img_cols[img_i % img_per_row]:
                            try:
                                pil_img = Image.open(BytesIO(img_data['data']))
                                if not stealth_mode:
                                    st.image(pil_img, use_container_width=True,
                                         caption=f"#{img_i+1} {img_data.get('key','')}")
                                else:
                                    st.caption(f"Image #{img_i+1}")
                                fname = (
                                    f"batch_{job_record['submitted_at'][:10]}_"
                                    f"{img_data.get('key','img')}_{img_i}.png"
                                ).replace(" ", "_").replace(":", "-")

                                # Convert to PNG bytes for download
                                dl_buf = BytesIO()
                                pil_img.save(dl_buf, format="PNG")
                                st.download_button(
                                    "📥",
                                    data=dl_buf.getvalue(),
                                    file_name=fname,
                                    mime="image/png",
                                    key=f"google_batch_dl_{job_idx}_{img_i}",
                                    use_container_width=True
                                )
                            except Exception:
                                st.warning(f"Could not render image #{img_i+1}")

                    # Bulk download as zip
                    if len(result_images) > 1:
                        import zipfile, io as _io
                        zip_buf = _io.BytesIO()
                        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for img_i, img_data in enumerate(result_images):
                                try:
                                    pil_img = Image.open(BytesIO(img_data['data']))
                                    png_buf = BytesIO()
                                    pil_img.save(png_buf, format="PNG")
                                    zf.writestr(
                                        f"batch_{img_data.get('key','img')}_{img_i}.png",
                                        png_buf.getvalue()
                                    )
                                except Exception:
                                    pass
                        zip_buf.seek(0)
                        zip_name = (
                            f"batch_{job_record['submitted_at'][:10]}_all.zip"
                        ).replace(" ", "_").replace(":", "-")
                        st.download_button(
                            f"📦 Download all {len(result_images)} images as ZIP",
                            data=zip_buf.getvalue(),
                            file_name=zip_name,
                            mime="application/zip",
                            key=f"google_batch_dl_zip_{job_idx}",
                            use_container_width=True
                        )

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

"""
Ecco il quadro aggiornato al marzo 2026, basato sulla documentazione ufficiale e sulle ultime notizie:

---

## 🆓 Free Tier Google AI Studio — Limiti per generazione immagini

### Modelli **Gemini** (image generation via `generate_content`)

| Modello | RPM | RPD (al giorno) |
|---|---|---|
| **gemini-2.5-flash-image** | ~10 | ~20–50 ⚠️ |
| **gemini-3-pro-image-preview** (Nano Banana Pro) | ~10 | ~20–100 ⚠️ |
| **gemini-3.1-flash-image-preview** (Nano Banana Flash) | ~10 | ~20–100 ⚠️ |

⚠️ A dicembre 2025 Google ha tagliato drasticamente i limiti free: Gemini 2.5 Flash è passato da ~250 richieste/giorno a circa 20, e Gemini 2.5 Pro è stato rimosso del tutto dal free tier per molti account. I modelli "Nano Banana" (Gemini 3.x image) sono in **preview**, e per i modelli in preview i limiti free sono approssimativamente 10–50 RPM e 100+ RPD, ma molti utenti riportano limiti effettivi molto più restrittivi.

---

### 💡 Consiglio pratico

Attenzione: quando attivi il billing su un progetto Google AI, il free tier viene completamente rimosso — ogni chiamata API diventa a pagamento. Non esiste un sistema ibrido "X richieste gratis poi a pagamento".

Per vedere i tuoi limiti **esatti e aggiornati** in tempo reale, vai su: **[aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit)** — la pagina mostra i limiti specifici del tuo progetto e tier corrente.


Ecco le tabelle in markdown:

---

### Gemini (Nano Banana) — via `generate_content`

| Modello | Model ID | Free tier | Costo/img (paid) | Note |
|---|---|---|---|---|
| Gemini 2.5 Flash Image | `gemini-2.5-flash-image` | ✅ Disponibile | ~$0.039 (1024px) | $30/1M token · 1290 token/img |
| Gemini 3 Pro Image | `gemini-3-pro-image-preview` | ❌ No | ~$0.067 (1024px) | $60/1M token · 1120 token/img |
| Gemini 3.1 Flash Image | `gemini-3.1-flash-image-preview` | ❌ No | ~$0.067 (1024px) | $60/1M token · 1120 token/img |

---

### Imagen 4 — via `generate_images`

| Modello | Model ID | Free tier | Costo/img (paid) | Note |
|---|---|---|---|---|
| Imagen 4 Standard | `imagen-4.0-generate-001` | ❌ No | $0.04 | Qualità massima |
| Imagen 4 Ultra | `imagen-4.0-ultra-generate-001` | ❌ No | $0.06 | Preview, qualità ultra |
| Imagen 4 Fast | `imagen-4.0-fast-generate-001` | ❌ No | $0.02 | Più economico |

---

### Costo per risoluzione — modelli Gemini

| Risoluzione | Token | Gemini 2.5 Flash ($30/1M) | Gemini 3.x ($60/1M) |
|---|---|---|---|
| 512px | 747 | ~$0.022 | ~$0.045 |
| 1024px | 1120–1290 | ~$0.039 | ~$0.067 |
| 2048px | 1680 | ~$0.050 | ~$0.101 |
| 4096px | 2520 | ~$0.076 | ~$0.151 |

"""
