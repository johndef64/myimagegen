#%%
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image, ImageOps
import json
import os
from datetime import datetime
import asyncio


# Instructions and settings
json_example = """{
{
  "subject": {
    "description": "A young woman taking a mirror selfie with very long voluminous dark waves and soft wispy bangs",
    "age": "young adult",
    "expression": "confident and slightly playful",
    "hair": {
      "color": "dark",
      "style": "very long, voluminous waves with soft wispy bangs"
    },
    "clothing": {
      "top": {
        "type": "fitted cropped t-shirt",
        "color": "cream white",
        "details": "features a large cute anime-style cat face graphic with big blue eyes, whiskers, and a small pink mouth"
      }
    },
    "face": {
      "preserve_original": true,
      "makeup": "natural glam makeup with soft pink dewy blush and glossy red pouty lips"
    }
  },
  "accessories": {
    "earrings": {
      "type": "gold geometric hoop earrings"
    },
    "jewelry": {
      "waistchain": "silver waistchain"
    },
    "device": {
      "type": "smartphone",
      "details": "patterned case"
    }
  },
  "photography": {
    "camera_style": "early-2000s digital camera aesthetic",
    "lighting": "harsh super-flash with bright blown-out highlights but subject still visible",
    "angle": "mirror selfie",
    "shot_type": "tight selfie composition",
    "texture": "subtle grain, retro highlights, V6 realism, crisp details, soft shadows"
  },
  "background": {
    "setting": "nostalgic early-2000s bedroom",
    "wall_color": "pastel tones",
    "elements": [
      "chunky wooden dresser",
      "CD player",
      "posters of 2000s pop icons",
      "hanging beaded door curtain",
      "cluttered vanity with lip glosses"
    ],
    "atmosphere": "authentic 2000s nostalgic vibe",
    "lighting": "retro"
  }
}
  """

INSTUCTIONS = {
    "GENERATE_TAGS": "Your task is to generate general descriptive tags for the image. Provide a comma separated list of relevant keywords that capture the main elements, themes, and subjects present in the image. Answer only with the tags, without any additional explanation or description.",
    "GENERATE_DETAILED_TAGS": "Your task is to generate a detailed list of descriptive tags for the image. Provide a comprehensive comma separated list of relevant keywords that capture the main elements, themes, subjects, colors, and notable features present in the image. Answer only with the detailed tags, without any additional explanation or description.",
    "GENERATE_SDXL_PROMPT": "Your task is to generate a concise textual prompt from this image/photo. the textual prompt must be suitable for image generation models base on SDXL, Illustrious. Use a comma separated format of relevant keywords that capture the main elements, themes, and subjects present in the image. Answer only with the concise prompt, without any additional explanation or description.",
    "GENERATE_PROMPT": "Your task is to write the textual prompt from this image/photo. the textual prompt must be suitable for image generation models. Answer only with the prompt, without any additional explanation or description.",
    "GENERATE_DETAILED_PROMPT": "Your task is to write a highly detailed textual prompt from this image/photo. the textual prompt must be suitable for image generation models. Include specific details about the scene, subjects, colors, lighting, and any notable features. Answer only with the detailed prompt, without any additional explanation or description.",
    "GENERATE_JSON_PROMPT": f"Your task is to write a JSON-formatted prompt from this image/photo. The JSON should include fields such as 'description', 'style', 'colors', 'face', 'makeup' and 'clothing' that capture the essence of the image. Answer only with the JSON object, without any additional explanation or description. Promt example: {json_example}",
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
    "grok-3-mini": "x-ai/grok-3-mini",
    "grok-3":  "x-ai/grok-3",
    "grok-4":  "x-ai/grok-4",
    "grok-4.1-fast":  "x-ai/grok-4.1-fast",

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

ALL_MODELS = {
    "groq": GROQ_MODELS,
    "grok": XAI_MODELS,
    "openrouter": OPENROUTER_MODELS,
}

def resolve_model(model_name):
    """Resolve a model name (key or value) to (provider, full_model_id).

    Accepts either a friendly key (e.g. "grok-4") or the full model ID
    (e.g. "x-ai/grok-4"). Raises ValueError with available models if not found.
    """
    for provider, models in ALL_MODELS.items():
        # Match by key → return value
        if model_name in models:
            return provider, models[model_name]
        # Match by value directly
        if model_name in models.values():
            return provider, model_name
    # Fallback: assume OpenAI model if it looks like a known pattern
    if model_name.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai", model_name
    # Try to guess provider from model ID prefix (e.g. "x-ai/..." → openrouter)
    prefix_to_provider = {
        "x-ai/": "openrouter", "meta-llama/": "groq", "openai/": "groq",
        "moonshotai/": "groq", "perplexity/": "openrouter", "openrouter/": "openrouter",
    }
    for prefix, provider in prefix_to_provider.items():
        if model_name.startswith(prefix):
            import warnings
            warnings.warn(f"Model '{model_name}' not in known list, guessing provider '{provider}'")
            return provider, model_name
    # Not found → helpful error
    all_names = []
    for provider, models in ALL_MODELS.items():
        for key, value in models.items():
            all_names.append(f"  {key:25s} → {value} ({provider})")
    raise ValueError(
        f"Model '{model_name}' not found. Available models:\n" + "\n".join(all_names)
    )

def select_client_based_on_model(model_name, verbose=False):
    """Select appropriate client based on model name"""
    provider, _ = resolve_model(model_name)
    if verbose:
        print(f"Using {provider} client")
    return get_client(provider)

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
    """Streamlit-compatible TaggerGPT class.

    Usage:
        tagger = TaggerGPT("grok-4")          # friendly key
        tagger = TaggerGPT("x-ai/grok-4")     # full model ID
        tagger = TaggerGPT("gpt-4o")           # OpenAI model
    """
    def __init__(self, model_name):
        provider, resolved = resolve_model(model_name)
        self.model_name = resolved
        self.client = get_client(provider)
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
    
    # async versione che processa una lista di immagini in parallelo con lo stesso prompt
    async def _chat_completion_prompt_single(self, system_prompt, user_prompt, image):
        """Async wrapper for single image processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat_completion_prompt, system_prompt, user_prompt, image)
    
    async def chat_completion_prompt_async(self, system_prompt, user_prompt, images):
        """Process multiple images in parallel with the same prompt."""
        tasks = []
        for image in images:
            tasks.append(self._chat_completion_prompt_single(system_prompt, user_prompt, image))
        return await asyncio.gather(*tasks)

# %% Test async processing in Jupyter
# In Jupyter, puoi usare await direttamente nella cella
if __name__ == "__main__":
    import glob
    import random

    tagger = TaggerGPT("grok-4")
    image_paths = glob.glob("../images/fem/bellezze/*.jpg")
    image_paths = random.sample(image_paths, 20)  
    images = [Image.open(path) for path in image_paths][:2]
    system_prompt = INSTUCTIONS["GENERATE_SDXL_PROMPT"]
    response = tagger.chat_completion_prompt(system_prompt, "Generate a concise prompt for this image.", images[0])
    print(response, "\n" + "-" * 50)
    time_start = datetime.now()
    _coro = tagger.chat_completion_prompt_async(system_prompt, "Generate a concise prompt for this image.", images)
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        _loop = None
    if _loop and _loop.is_running():
        # Jupyter: event loop già attivo → usa nest_asyncio o await diretto
        import nest_asyncio
        nest_asyncio.apply()
        responses = _loop.run_until_complete(_coro)
    else:
        # Script normale
        responses = asyncio.run(_coro)
    time_end = datetime.now()
    print(f"Elapsed time: {time_end - time_start}")
    print(f"Average time per image: {(time_end - time_start) / len(images)}")

    # salva le risposte in un file facile da reimportare come lista
    with open("text_sdxl_prompts.txt", "w") as f:
        for res in responses:
            f.write(res + "\n")

    for res in responses:
        print(res)
        print("-" * 50)


# %%