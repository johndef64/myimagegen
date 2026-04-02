#%%
"""
ModelsLab API Client - Clean and modular interface for ModelsLab image generation API.

This module provides:
- ModelsLabAPI: Main class for managing API requests, results, and saving
- Independent utility functions for image/file management (can be used standalone)

Supported endpoints:
- txt2img: Text to image generation
- img2img: Image to image transformation  
- qwen-edit: Qwen model image editing
- v7 img2img: Advanced image to image (seedream, gen4, etc.)

Author: Refactored from mlslab_utils.py
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

class Endpoint(Enum):
    """Supported API endpoints."""
    TXT2IMG = "text2img"
    IMG2IMG = "img2img"
    QWEN_EDIT = "qwen_edit"
    IMG2IMG_V7 = "img2img_v7"
    TXT2IMG_V7 = "text2img_v7"


class AspectRatio(Enum):
    """Standard aspect ratios with corresponding pixel dimensions (~1MP)."""
    SQUARE = "1:1"
    PORTRAIT_3_4 = "3:4"
    PORTRAIT_1_2 = "1:2"
    PORTRAIT_9_16 = "9:16"
    PORTRAIT_2_3 = "2:3"
    LANDSCAPE_4_3 = "4:3"
    LANDSCAPE_2_1 = "2:1"
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_3_2 = "3:2"


# Dimension maps for different endpoints
SIZE_IMAGE_DICT = {
    "1:1": (1024, 1024),
    "3:4": (888, 1184),
    "1:2": (728, 1456),
    "9:16": (768, 1360),
    "4:3": (1184, 888),
    "2:1": (1456, 728),
    "16:9": (1360, 768),
    "2:3": (888, 1184),
    "3:2": (1184, 888),
}

QWEN_SIZE_DICT = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

SCHEDULER_LIST = [
    "DDPMScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler",
    "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "DPMSolverSinglestepScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "UniPCMultistepScheduler",
    "DDIMInverseScheduler",
    "DEISMultistepScheduler",
    "IPNDMScheduler",
    "KarrasVeScheduler",
    "ScoreSdeVeScheduler",
    "LCMScheduler",
]

FLUXDEV_LORAS = {
    "Blindbox Flux Lora V2.0": "blindbox-flux-lora-v2-0",
    "Flux Dev Aesthetics Upgrade V1.0": "flux-dev-aesthetics-upgrade-lora-v1-0",
    "Real Lora V2.0 (Realism)": "real-lora-v2-0",
    "Long Hair LoRA Flux V2": "long-hair-lora-flux-v2",
    "Uncensored Flux Lora": "specialized-for-unrestricted-detailed-generation",
    "Flux NSFW Lora V2": "sldr_flux_nsfw_v2",
    "Fc Anime Lora Flux": "fc-anime-lora-flux-fcanimeflux",
    "Urban Collage Style Flux Dev": "urban-collage-style-flux-dev-lora-v1-0",
    "Flux Krea Realism LoRA V1.0": "flux-krea-realism-lora-v1-0",
    "Flux Dev to Schnell 4 Step": "flux-dev-to-schnell-4-step-lora-bf16",
    "Flux Lora Collection Xlabs": "flux-lora-collection",
    "UltraRealistic Lora Project": "ultrarealistic-lora-project", 
}


# ============================================================================
# MODEL CAPABILITIES & CONFIGURATIONS
# ============================================================================
# Key insights:
# - init_image can always be a list, endpoint handles distribution to init_image_1, init_image_2, etc.
#   EXCEPT: flux-kontext-dev requires init_image as string (single image only)
# - For img2img/qwen_edit: omit height/width to use input image dimensions
# - All V6/V7 models support both txt2img and img2img EXCEPT flux-kontext-dev (img2img only)
# - flux models (except flux-kontext-dev) require scheduler field for img2img
# - Send minimal payload by default, only add parameters when explicitly set

class ModelCapability:
    """Constants for model capabilities."""
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    QWEN_EDIT = "qwen_edit"

# V6 models - support both txt2img and img2img (except noted)
V6_MODELS = [
    "qwen", "z-image-turbo", "z-image-base", "flux", "fluxdev", 
    "flux-2-dev", "flux-kontext-dev", "flux-klein"
]

# V7 models (use different endpoints)
V7_MODELS = [
    "grok-imagine-image-t2i", "grok-imagine-image-i2i",
    "seedream-4.0-i2i", "gen4_image_turbo", "flux-2-pro", "nano-banana",
    "wan-2.7-i2i", "wan-2.7-t2i"
]

# Qwen edit models 
QWEN_EDIT_MODELS = ["qwen-edit", "qwen-edit-2511"]

# Models that require init_image as STRING (not list)
INIT_IMAGE_AS_STRING_MODELS = ["flux-kontext-dev"]

# Models that ONLY support img2img (no txt2img)
IMG2IMG_ONLY_MODELS = ["flux-kontext-dev"]

# Models that require scheduler field for img2img
SCHEDULER_REQUIRED_FOR_IMG2IMG = ["flux", "fluxdev", "flux-2-dev", "flux-klein"]

# Model configurations with default parameters
MODEL_CONFIGS = {
    # === Qwen Models ===
    "qwen": {
        "num_inference_steps": 8,
        "strength": 0.5,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "qwen-edit": {
        "num_inference_steps": 8,
        "api_version": "v6",
        "endpoint_img2img": Endpoint.QWEN_EDIT,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "qwen-edit-2511": {
        "num_inference_steps": 8,
        "api_version": "v6",
        "endpoint_img2img": Endpoint.QWEN_EDIT,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    # === Flux Models ===
    "flux-kontext-dev": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": False,  # Only img2img supported
        "supports_img2img": True,
        "init_image_as_list": False,  # MUST be string, not list
        "requires_scheduler": False,
    },
    "flux": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
        "requires_scheduler": True,  # Required for img2img
    },
    "fluxdev": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
        "requires_scheduler": True,  # Required for img2img
    },
    "flux-klein":    {
        "num_inference_steps": 28,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
        "requires_scheduler": True,  # Required for img2img
    },

    "flux-2-dev": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
        "requires_scheduler": True,  # Required for img2img
    },
    # === Z-Image Models ===
    "z-image-base": {
        "num_inference_steps": 20,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "z-image-turbo": {
        "num_inference_steps": 8,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    # === V7 Models ===
    "grok-imagine-image-t2i": {
        "api_version": "v7",
        "endpoint_txt2img": Endpoint.TXT2IMG_V7,  # V7 has separate URL
        "supports_txt2img": True,
        "supports_img2img": False,
        "init_image_as_list": True,
    },
    "grok-imagine-image-i2i": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "seedream-4.0-i2i": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "gen4_image_turbo": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "flux-2-pro": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "nano-banana": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    # === WAN 2.7 Models ===
    "wan-2.7-i2i": {
        "api_version": "v7",
        "endpoint_img2img": Endpoint.IMG2IMG_V7,
        "supports_txt2img": False,
        "supports_img2img": True,
        "init_image_as_list": True,
    },
    "wan-2.7-t2i": {
        "api_version": "v7",
        "endpoint_txt2img": Endpoint.TXT2IMG_V7,
        "supports_txt2img": True,
        "supports_img2img": False,
        "init_image_as_list": True,
    },
}


def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get model configuration, returning defaults for unknown models."""
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    
    # Default config for unknown V6 models
    return {
        "num_inference_steps": 20,
        "strength": 0.7,
        "api_version": "v6",
        "endpoint_txt2img": Endpoint.TXT2IMG,
        "endpoint_img2img": Endpoint.IMG2IMG,
        "supports_txt2img": True,
        "supports_img2img": True,
        "init_image_as_list": True,
    }


def model_supports_txt2img(model_id: str) -> bool:
    """Check if model supports text-to-image generation."""
    config = get_model_config(model_id)
    return config.get("supports_txt2img", True)


def model_supports_img2img(model_id: str) -> bool:
    """Check if model supports image-to-image transformation."""
    config = get_model_config(model_id)
    return config.get("supports_img2img", True)


def model_requires_init_image_as_string(model_id: str) -> bool:
    """Check if model requires init_image as string (not list)."""
    config = get_model_config(model_id)
    return not config.get("init_image_as_list", True)


def model_requires_scheduler(model_id: str) -> bool:
    """Check if model requires scheduler for img2img."""
    config = get_model_config(model_id)
    return config.get("requires_scheduler", False)


def get_model_endpoint(model_id: str, has_images: bool = False) -> Endpoint:
    """Get the appropriate endpoint for a model based on operation type."""
    config = get_model_config(model_id)
    
    if has_images:
        return config.get("endpoint_img2img", Endpoint.IMG2IMG)
    else:
        return config.get("endpoint_txt2img", Endpoint.TXT2IMG)


# ============================================================================
# STANDALONE UTILITY FUNCTIONS (Image and File Management)
# ============================================================================

def _import_resize_function():
    """Import resize function handling both package and standalone imports."""
    try:
        from .image_params import resize_image_to_megapixels
    except ImportError:
        from image_params import resize_image_to_megapixels
    return resize_image_to_megapixels


def encode_image_to_base64(image_path: str, resize: Optional[float] = None) -> str:
    """
    Convert a local image to base64 string.
    
    Args:
        image_path: Path to the local image file
        resize: Optional target megapixels for resizing (e.g., 1.0 for 1MP)
    
    Returns:
        Base64 encoded string of the image
    """
    if resize:
        resize_image_to_megapixels = _import_resize_function()
        _, _, resized_img = resize_image_to_megapixels(image_path, target_mp=resize)
        buffered = BytesIO()
        resized_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_to_base64_with_prefix(image_path: str) -> str:
    """Convert a local image to base64 with data URI prefix."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"


def decode_base64_to_image(base64_data: str) -> bytes:
    """
    Decode base64 string to image bytes.
    
    Args:
        base64_data: Base64 encoded string (with or without data URI prefix)
    
    Returns:
        Image bytes
    """
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    return base64.b64decode(base64_data)


def is_url(path: str) -> bool:
    """Check if a string is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def is_base64(data: str) -> bool:
    """Check if a string is base64 encoded image data."""
    return data.startswith("data:image") or data.startswith("/9j/") or data.endswith(".base64")


def get_images_paths(folder: str, handle: Union[str, List[str]] = "") -> List[str]:
    """
    Get list of image file paths from a folder.
    
    Args:
        folder: Folder path to search
        handle: Optional filter string or list of strings to match in filename
    
    Returns:
        List of matching image file paths
    """
    import glob
    if isinstance(handle, list):
        paths = []
        for h in handle:
            paths.extend(glob.glob(f"{folder}/*{h}*"))
        return paths
    else:
        return glob.glob(f"{folder}/*{handle}*")


def show_folder_images_thumbnails(
    folder_path: str,
    max_images: Optional[int] = None,
    thumb_size: tuple = (100, 100)
) -> None:
    """Display thumbnails of images in a folder (for Jupyter notebooks)."""
    from IPython.display import display, Image as IPyImage
    
    if not max_images:
        max_images = len(os.listdir(folder_path))
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    count = 0
    
    for filename in os.listdir(folder_path):
        if count >= max_images:
            break
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(folder_path, filename)
            display(IPyImage(filename=filepath, width=thumb_size[0], height=thumb_size[1]))
            print(filename)
            count += 1


def show_image_thumbnail(
    img_source: str,
    size: tuple = (100, 100)
) -> None:
    """
    Display an image thumbnail from various sources.
    
    Args:
        img_source: URL, base64 URL, or local file path
        size: Thumbnail dimensions (width, height)
    """
    from IPython.display import display, Image as IPyImage
    
    if not img_source:
        print("Empty image source, cannot display.")
        return
    
    if is_url(img_source):
        # Check if it's a base64 URL (returns base64 text)
        if ".base64" in img_source:
            response = requests.get(img_source)
            response.raise_for_status()
            img_data = decode_base64_to_image(response.text)
            display(IPyImage(data=img_data, width=size[0], height=size[1]))
        else:
            display(IPyImage(url=img_source, width=size[0], height=size[1]))
    else:
        display(IPyImage(filename=img_source, width=size[0], height=size[1]))


from PIL import Image, ImageOps, PngImagePlugin
def save_image_with_metadata(image: Image.Image, prompt: str, model_name: str, 
                             seed: int, aspect_ratio: str, request_id: str = "",
                             output_folder: str = "outputs") -> str:
    """Save image with metadata to outputs folder"""
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Model", model_name)
    metadata.add_text("Seed", str(seed))
    metadata.add_text("Aspect_Ratio", aspect_ratio)
    metadata.add_text("Request_ID", request_id)
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


def save_image_from_base64(
    base64_data: str,
    filepath: str
) -> str:
    """
    Save base64 encoded image to a file.
    
    Args:
        base64_data: Base64 encoded image data
        filepath: Destination file path
    
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_data = decode_base64_to_image(base64_data)
    
    with open(filepath, "wb") as img_file:
        img_file.write(img_data)
    
    return filepath


def save_image_from_url(
    url: str,
    filepath: str,
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """
    Download and save an image from URL.
    
    Args:
        url: Image URL (can be direct image or base64 URL)
        filepath: Destination file path
    
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Check if response is base64
    if url.endswith(".base64") or response.headers.get("content-type", "").startswith("text"):
        print("Saving from base64 data...")
        img_data = decode_base64_to_image(response.text)
    else:
        img_data = response.content
    
    if metadata and filepath.lower().endswith(".png"):
        image = Image.open(BytesIO(img_data))
        png_metadata = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            png_metadata.add_text(key, value)
        image.save(filepath, pnginfo=png_metadata)
        return filepath
    else:
        with open(filepath, "wb") as img_file:
            img_file.write(img_data)
        
    return filepath


def clean_filenames_in_folder(folder: str, replace_spaces: bool = True) -> None:
    """Replace spaces with underscores in filenames within a folder."""
    for filename in os.listdir(folder):
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_filename)
            )



# ============================================================================
# DATA CLASSES FOR API RESPONSES
# ============================================================================

@dataclass
class APIResponse:
    """Structured API response data."""
    request_id: str
    status: str
    raw_response: Dict[str, Any]
    output_urls: List[str] = field(default_factory=list)
    future_links: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIResponse":
        """Create APIResponse from API response dictionary."""
        return cls(
            request_id=data.get("id", ""),
            status=data.get("status", "unknown"),
            raw_response=data,
            output_urls=data.get("output", []),
            future_links=data.get("future_links", []),
            error_message=data.get("message") if data.get("status") == "error" else None
        )
    
    @property
    def is_processing(self) -> bool:
        return self.status == "processing"
    
    @property
    def is_success(self) -> bool:
        return self.status == "success"
    
    @property
    def is_error(self) -> bool:
        return self.status == "error"


# ============================================================================
# PAYLOAD BUILDERS
# ============================================================================

class PayloadBuilder:
    """
    Smart payload builder for ModelsLab API endpoints.
    
    Key principles:
    - Send MINIMAL payload by default (only required parameters)
    - Parameters start as None; only add to payload when explicitly set
    - init_image can be a list (endpoint handles distribution to init_image_1, init_image_2, etc.)
      EXCEPT: flux-kontext-dev requires init_image as string
    - Omit height/width for img2img/qwen_edit to use input image dimensions
    - Add scheduler for flux models (except flux-kontext-dev) when doing img2img
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def _add_if_set(self, payload: Dict[str, Any], key: str, value: Any, 
                    stringify: bool = False) -> None:
        """Add key to payload only if value is not None."""
        if value is not None:
            payload[key] = str(value) if stringify else value
    
    def _prepare_init_image(self, images: List[str], model_id: str) -> Union[str, List[str]]:
        """
        Prepare init_image for payload based on model requirements.
        
        - Most models: init_image can be a list (endpoint distributes to init_image_1, etc.)
        - flux-kontext-dev: MUST be a string (single image only)
        """
        if not images:
            return None
        
        # Check if model requires string instead of list
        if model_requires_init_image_as_string(model_id):
            if len(images) > 1:
                print(f"Warning: {model_id} only supports single image, using first image only")
            return images[0]
        
        # Return list if multiple images, string if single (endpoint handles both)
        return images if len(images) > 1 else images[0]
    
    def build_txt2img_payload(
        self,
        prompt: str,
        model_id: str = "flux-2-dev",
        # All optional parameters default to None - only added when set
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scheduler: Optional[str] = None,
        enhance_prompt: Optional[str] = None,
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build MINIMAL payload for text2img endpoint.
        Only required: key, prompt, model_id
        Optional parameters are only added when explicitly set.
        """
        # Minimum required payload
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "model_id": model_id,
        }
        
        # Add optional parameters only if set
        self._add_if_set(payload, "width", width)
        self._add_if_set(payload, "height", height)
        self._add_if_set(payload, "seed", seed)
        self._add_if_set(payload, "negative_prompt", negative_prompt)
        self._add_if_set(payload, "num_inference_steps", num_inference_steps, stringify=True)
        self._add_if_set(payload, "guidance_scale", guidance_scale, stringify=True)
        self._add_if_set(payload, "scheduler", scheduler)
        self._add_if_set(payload, "enhance_prompt", enhance_prompt)
        self._add_if_set(payload, "lora_model", lora_model)
        self._add_if_set(payload, "lora_strength", lora_strength, stringify=True)
        self._add_if_set(payload, "samples", samples, stringify=True)
        
        # Add any extra kwargs
        payload.update(kwargs)
        return payload
    
    def build_img2img_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "flux-2-dev",
        # Optional - omit to use input image dimensions
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scheduler: Optional[str] = None,
        strength: Optional[float] = None,
        enhance_prompt: Optional[str] = None,
        use_base64: Optional[str] = None,
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build MINIMAL payload for img2img endpoint.
        
        Key behaviors:
        - init_image can be list (endpoint distributes to init_image_1, etc.)
          EXCEPT flux-kontext-dev which requires string
        - Omit width/height to use input image dimensions
        - Scheduler is auto-added for flux models (except flux-kontext-dev)
        """
        # Prepare init_image based on model requirements
        init_image = self._prepare_init_image(images, model_id)
        
        # Minimum required payload
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "model_id": model_id,
            "init_image": init_image,
        }
        
        # Add width/height only if explicitly set (otherwise uses input image dims)
        self._add_if_set(payload, "width", width)
        self._add_if_set(payload, "height", height)
        
        # Auto-add scheduler for flux models that require it
        if model_requires_scheduler(model_id):
            if scheduler is None:
                scheduler = "DPMSolverMultistepScheduler"  # Default scheduler
            payload["scheduler"] = scheduler
        elif scheduler is not None:
            payload["scheduler"] = scheduler
        
        # Add other optional parameters only if set
        self._add_if_set(payload, "seed", seed)
        self._add_if_set(payload, "negative_prompt", negative_prompt)
        self._add_if_set(payload, "num_inference_steps", num_inference_steps, stringify=True)
        self._add_if_set(payload, "guidance_scale", guidance_scale, stringify=True)
        self._add_if_set(payload, "strength", strength, stringify=True)
        self._add_if_set(payload, "enhance_prompt", enhance_prompt)
        self._add_if_set(payload, "base64", use_base64)
        self._add_if_set(payload, "lora_model", lora_model)
        self._add_if_set(payload, "lora_strength", lora_strength, stringify=True)
        self._add_if_set(payload, "samples", samples, stringify=True)
        
        # Add any extra kwargs
        payload.update(kwargs)
        return payload
    
    def build_qwen_edit_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "qwen-edit",
        # Optional - omit to use input image dimensions
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        use_base64: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build MINIMAL payload for qwen_edit endpoint.
        
        - init_image can be a list (up to 4 images supported)
        - Omit width/height to use input image dimensions
        """
        # For qwen_edit, init_image can always be a list
        init_image = images if len(images) > 1 else images[0] if images else None
        
        # Minimum required payload
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "init_image": init_image,
        }
        
        # Only add model_id if not default qwen-edit
        if model_id and model_id != "qwen-edit":
            payload["model_id"] = model_id
        
        # Add width/height only if explicitly set (otherwise uses input image dims)
        self._add_if_set(payload, "width", width)
        self._add_if_set(payload, "height", height)


        # Add other optional parameters
        self._add_if_set(payload, "seed", seed)
        self._add_if_set(payload, "base64", use_base64)
        self._add_if_set(payload, "num_inference_steps", num_inference_steps, stringify=True)
        
        # Add any extra kwargs
        payload.update(kwargs)
        
        return payload
    
    def build_img2img_v7_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "seedream-4.0-i2i",
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build MINIMAL payload for v7 img2img endpoint.
        
        V7 uses aspect_ratio and resolution instead of width/height.
        init_image can be a list.
        """
        # For v7, init_image can be a list
        init_image = images if len(images) > 1 else images[0] if images else None

        # Detect if any image is base64 (not a URL)
        has_base64 = any(not is_url(img) for img in images) if images else False

        # Minimum required payload
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "model_id": model_id,
            "init_image": init_image,
            "safety_checker": "no",
        }

        if has_base64:
            payload["base64"] = "yes"
        
        # Add optional parameters
        if aspect_ratio:
            valid_ratios = ["1:1", "4:3", "9:16", "16:9", "3:2", "2:3", "21:9", "9:21"]
            payload["aspect_ratio"] = aspect_ratio if aspect_ratio in valid_ratios else "1:1"
        
        self._add_if_set(payload, "resolution", resolution)
        self._add_if_set(payload, "seed", seed)
        
        # Add any extra kwargs
        payload.update(kwargs)
        return payload
    
    def build_txt2img_v7_payload(
        self,
        prompt: str,
        model_id: str = "grok-imagine-image-t2i",
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build MINIMAL payload for v7 txt2img endpoint.
        """
        # Minimum required payload
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "model_id": model_id,
        }
        
        # Add optional parameters
        if aspect_ratio:
            valid_ratios = ["1:1", "4:3", "9:16", "16:9", "3:2", "2:3", "21:9", "9:21"]
            payload["aspect_ratio"] = aspect_ratio if aspect_ratio in valid_ratios else "1:1"
        
        self._add_if_set(payload, "resolution", resolution)
        self._add_if_set(payload, "seed", seed)
        
        # Add any extra kwargs  
        payload.update(kwargs)
        return payload


# Legacy method removed - no longer needed
# def build_base_payload was folded into individual build methods


# ============================================================================
# MAIN API CLASS
# ============================================================================

class ModelsLabAPI:
    """
    Main class for interacting with ModelsLab API.
    
    Handles:
    - API requests (txt2img, img2img, qwen-edit, v7)
    - Result fetching and polling
    - Saving outputs (images, URLs, logs)
    - Retry logic
    
    Example usage:
        api = ModelsLabAPI(api_key="your_key")
        
        # Text to image
        response = api.txt2img("A beautiful sunset")
        
        # Image to image
        response = api.img2img(
            prompt="Transform to anime style",
            images=["path/to/image.jpg"]
        )
        
        # Wait for result and save
        result = api.wait_for_result(response.request_id)
        api.save_result(result, folder="outputs")
    """
    
    # API URLs
    URLS = {
        Endpoint.TXT2IMG: "https://modelslab.com/api/v6/images/text2img",
        Endpoint.IMG2IMG: "https://modelslab.com/api/v6/images/img2img",
        Endpoint.QWEN_EDIT: "https://modelslab.com/api/v6/image_editing/qwen_edit",
        Endpoint.IMG2IMG_V7: "https://modelslab.com/api/v7/images/image-to-image",
        Endpoint.TXT2IMG_V7: "https://modelslab.com/api/v7/images/text-to-image",
    }
    
    FETCH_URL = "https://modelslab.com/api/v6/images/fetch"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_path: str = "../api_keys.json",
        output_folder: str = "outputs",
        requests_log_file: str = "requests_list.txt",
        urls_log_file: str = "edited_image_urls.txt",
        default_resize_mp: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize ModelsLab API client.
        
        Args:
            api_key: API key (if None, loads from api_key_path)
            api_key_path: Path to JSON file containing API key
            output_folder: Default folder for saving outputs
            requests_log_file: File to log request IDs
            urls_log_file: File to log output URLs
            default_resize_mp: Default megapixels for image resizing
            verbose: Whether to print status messages
        """
        self.api_key = api_key or self._load_api_key(api_key_path)
        self.output_folder = output_folder
        self.requests_log_file = requests_log_file
        self.urls_log_file = urls_log_file
        self.default_resize_mp = default_resize_mp
        self.verbose = verbose
        
        self.payload_builder = PayloadBuilder(self.api_key)
        self._pending_requests: List[str] = []
        self._all_requests: List[str] = []  # All request IDs from this session
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        if self.verbose:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
            print(f"ModelsLabAPI initialized with key: {masked_key}")
    
    def _load_api_key(self, path: str) -> str:
        """Load API key from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            keys = json.load(f)
            return keys.get("modelslab", "")
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _prepare_images(
        self,
        images: Optional[Union[str, List[str]]],
        resize_mp: Optional[float] = None
    ) -> tuple:
        """
        Prepare images for API request (convert to base64 if needed).
        
        Args:
            images: Single image or list of images (paths or URLs)
            resize_mp: Target megapixels for resizing
        
        Returns:
            Tuple of (prepared_images, use_base64)
            - prepared_images: List of prepared image data (base64 or URLs)
            - use_base64: "yes" if images are base64, "no" if URLs
        """
        if images is None:
            return [], "yes"
        
        if isinstance(images, str):
            images = [images]
        
        resize = resize_mp or self.default_resize_mp
        prepared = []
        has_urls = False
        has_local = False
        
        for img in images:
            if is_url(img):
                # Keep URLs as-is
                prepared.append(img)
                has_urls = True
                self._log(f"  → Image is URL: {img[:60]}...")
            elif is_base64(img):
                # Already base64
                prepared.append(img)
                has_local = True
                self._log(f"  → Image is already base64")
            else:
                # Local file - convert to base64
                self._log(f"  → Converting local file to base64: {img}")
                b64 = encode_image_to_base64(img, resize=resize)
                prepared.append(b64)
                has_local = True
        
        # Determine base64 flag
        # If all images are URLs, use base64="no"
        # If any image is local/base64, use base64="yes"
        use_base64 = "no" if has_urls and not has_local else "yes"
        self._log(f"  → Using base64={use_base64}")
        
        return prepared, use_base64
    
    def _get_dimensions(
        self,
        images: Optional[List[str]],
        aspect_ratio: Optional[str] = None,
        size_dict: Dict[str, tuple] = SIZE_IMAGE_DICT,
        resize_mp: Optional[float] = None
    ) -> tuple:
        """
        Determine output dimensions based on aspect ratio or input image.
        
        Priority:
        1. If aspect_ratio is specified, use dimensions from size_dict
        2. If images provided, get dimensions from first image after resize
           (supports local files, URLs, and base64)
        3. Default to 1024x1024
        
        Args:
            images: Input images (for auto-detecting size)
            aspect_ratio: Desired aspect ratio (e.g., "16:9")
            size_dict: Dictionary mapping aspect ratios to dimensions
            resize_mp: Target megapixels for resizing (uses default if None)
        
        Returns:
            Tuple of (width, height)
        """
        # Priority 1: Use aspect ratio if specified
        if aspect_ratio and aspect_ratio in size_dict:
            self._log(f"Using dimensions for aspect ratio {aspect_ratio}: {size_dict[aspect_ratio]}")
            return size_dict[aspect_ratio]
        
        # Priority 2: Get dimensions from first image (works with local, URL, base64)
        if images:
            try:
                resize_image_to_megapixels = _import_resize_function()
                target_mp = resize_mp or self.default_resize_mp
                width, height, _ = resize_image_to_megapixels(
                    images[0], 
                    target_mp=target_mp
                )
                # Ensure divisible by 8
                width = width - (width % 8)
                height = height - (height % 8)
                self._log(f"Using dimensions from input image (resized to {target_mp}MP): {width}x{height}")
                return width, height
            except Exception as e:
                self._log(f"Failed to get dimensions from image: {e}")
        
        # Default
        self._log("Using default dimensions: 1024x1024")
        return 1024, 1024
    
    def _make_request(
        self,
        endpoint: Endpoint,
        payload: Dict[str, Any]
    ) -> APIResponse:
        """
        Make API request to specified endpoint.
        
        Args:
            endpoint: Target API endpoint
            payload: Request payload
        
        Returns:
            APIResponse object
        """
        url = self.URLS[endpoint]
        headers = {"Content-Type": "application/json"}
        
        self._log(f"Making request to {endpoint.value}...")

        
        response = requests.post(url, headers=headers, json=payload)
        if not response.ok:
            print(f"[ModelsLabAPI] HTTP {response.status_code} error body: {response.text}")
            response.reason = f"{response.reason} | {response.text}"
            response.raise_for_status()
        
        result = APIResponse.from_dict(response.json())
        
        # Save request ID to session list
        if result.request_id:
            self._all_requests.append(result.request_id)
        
        # Log the request
        self._log_request(result)
        
        if result.is_processing:
            self._pending_requests.append(result.request_id)
            self._log(f"✓ Request queued: {result.request_id}")
        elif result.is_error:
            self._log(f"✗ Request failed: {result.error_message}")
        
        return result
    
    def _log_request(self, response: APIResponse) -> None:
        """Log request ID to file."""
        log_path = os.path.join(self.output_folder, self.requests_log_file)
        
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("# ModelsLab API Requests Log\n")
        
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n# {timestamp}\n{response.request_id}\n")
    
    # ========== PUBLIC API METHODS ==========
    
    def txt2img(
        self,
        prompt: str,
        model_id: str = "flux-2-dev",
        aspect_ratio: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scheduler: Optional[str] = None,
        seed: Optional[int] = None,
        enhance_prompt: Optional[str] = None,
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        samples: Optional[int] = None,
        **kwargs
    ) -> APIResponse:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Text description of desired image
            model_id: Model to use (e.g., "flux-2-dev", "z-image-turbo")
            aspect_ratio: Output aspect ratio (e.g., "16:9")
            width: Output width (overrides aspect_ratio)
            height: Output height (overrides aspect_ratio)
            negative_prompt: What to avoid in generation
            num_inference_steps: Generation steps (None = model default)
            guidance_scale: CFG scale (None = API default)
            scheduler: Scheduler algorithm (None = API default)
            seed: Random seed (None = random)
            enhance_prompt: Whether to enhance prompt ("yes"/"no")
            lora_model: LoRA model ID to use
            lora_strength: LoRA strength (0.0-1.0)
            samples: Number of images to generate
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
        """
        # Check if model supports txt2img
        if not model_supports_txt2img(model_id):
            raise ValueError(f"Model '{model_id}' does not support txt2img. Use img2img instead.")
        
        # Get model config for defaults
        config = get_model_config(model_id)
        
        # Determine dimensions 
        w, h = None, None
        if width is not None and height is not None:
            w, h = width, height
        elif aspect_ratio:
            w, h = self._get_dimensions(None, aspect_ratio)
        # else: let API use defaults (don't include in payload)
        
        # Get defaults from model config (only if not explicitly set)
        if num_inference_steps is None:
            num_inference_steps = config.get("num_inference_steps")
        
        payload = self.payload_builder.build_txt2img_payload(
            prompt=prompt,
            model_id=model_id,
            width=w,
            height=h,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler=scheduler,
            enhance_prompt=enhance_prompt,
            lora_model=lora_model,
            lora_strength=lora_strength,
            samples=samples,
            **kwargs
        )
        
        # Determine correct endpoint based on model
        endpoint = get_model_endpoint(model_id, has_images=False)
        
        return self._make_request(endpoint, payload)
    
    def img2img(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "flux-2-dev",
        aspect_ratio: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scheduler: Optional[str] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        resize_mp: Optional[float] = None,
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        use_input_dimensions: bool = True,
        **kwargs
    ) -> APIResponse:
        """
        Transform existing image based on prompt.
        
        Args:
            prompt: Text description of desired transformation
            images: Input image(s) - paths or URLs
            model_id: Model to use (default: flux-2-dev)
            aspect_ratio: Output aspect ratio (sets width/height)
            width: Output width (None = use input image width)
            height: Output height (None = use input image height)
            negative_prompt: What to avoid
            num_inference_steps: Generation steps (None = use model default)
            guidance_scale: CFG scale (None = API default)
            scheduler: Scheduler algorithm (auto-added for flux models that require it)
            strength: How much to transform (None = use model default)
            seed: Random seed (None = random)
            resize_mp: Resize input to this megapixel size
            lora_model: LoRA model ID to use
            lora_strength: LoRA strength (0.0-1.0)
            use_input_dimensions: If True and no width/height specified, omit from payload
                                  to let the API use input image dimensions
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
            
        Notes:
            - init_image is passed as list (endpoint distributes to init_image_1, etc.)
              EXCEPT flux-kontext-dev which requires string (single image only)
            - Scheduler is auto-added for flux/fluxdev/flux-2-dev models
            - Omit width/height to use input image dimensions
        """
        # Check if model supports img2img
        if not model_supports_img2img(model_id):
            raise ValueError(f"Model '{model_id}' does not support img2img. Use txt2img instead.")
        
        # Determine resize target
        target_resize_mp = resize_mp or self.default_resize_mp
        
        # Get model config for defaults
        config = get_model_config(model_id)
        
        # Determine dimensions
        # Key insight: omit width/height from payload to use input image dimensions
        w, h = None, None
        
        if width is not None and height is not None:
            # User explicitly specified dimensions
            w, h = width, height
            self._log(f"Using specified dimensions: {w}x{h}")
        elif aspect_ratio:
            # User specified aspect ratio - calculate dimensions
            w, h = self._get_dimensions(None, aspect_ratio)
            self._log(f"Using aspect ratio {aspect_ratio}: {w}x{h}")
        elif not use_input_dimensions:
            # User wants explicit dimensions but didn't provide them
            # Get from input image
            original_images = [images] if isinstance(images, str) else images
            w, h = self._get_dimensions(original_images, None, resize_mp=target_resize_mp)
            self._log(f"Using dimensions from input image: {w}x{h}")
        else:
            # Let API use input image dimensions (don't include in payload)
            self._log("Omitting dimensions - API will use input image dimensions")
        
        # Prepare images and detect if using base64 or URLs
        prepared_images, use_base64 = self._prepare_images(images, target_resize_mp)
        
        # Get defaults from model config (only if not explicitly set)
        if num_inference_steps is None:
            num_inference_steps = config.get("num_inference_steps")
        if strength is None:
            strength = config.get("strength")
        
        # Build payload with minimal parameters
        payload = self.payload_builder.build_img2img_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            width=w,  # None means omit from payload
            height=h,  # None means omit from payload
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler=scheduler,  # PayloadBuilder handles auto-add for flux models
            strength=strength,
            use_base64=use_base64,
            lora_model=lora_model,
            lora_strength=lora_strength,
            **kwargs
        )
        
        # Determine correct endpoint based on model
        endpoint = get_model_endpoint(model_id, has_images=True)
        
        # make request with retry
        response = self._make_request(endpoint, payload)

        return response
    
    def qwen_edit(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "qwen-edit",
        aspect_ratio: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        resize_mp: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        use_input_dimensions: bool = True,
        **kwargs
    ) -> APIResponse:
        """
        Edit image using Qwen model.
        
        Args:
            prompt: Edit instructions
            images: Input image(s) - up to 4 images supported (passed as list)
            model_id: "qwen-edit" or "qwen-edit-2511"
            aspect_ratio: Output aspect ratio (sets width/height)
            width: Output width (None = use input image width)
            height: Output height (None = use input image height)
            seed: Random seed (None = random)
            resize_mp: Resize input to this megapixel size (default 1.7 for Qwen)
            num_inference_steps: Generation steps (None = model default)
            use_input_dimensions: If True and no width/height specified, omit from payload
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
            
        Notes:
            - init_image can be a list (up to 4 images for multi-image editing)
            - Omit width/height to use input image dimensions
        """
        # Determine resize target (Qwen supports higher resolution)
        target_resize = resize_mp or 1.7
        
        # Get model config for defaults
        config = get_model_config(model_id)
        
        # Determine dimensions
        w, h = None, None

        if model_id == "qwen-edit-2511":
            use_input_dimensions = False
        
        if width is not None and height is not None:
            w, h = width, height
            self._log(f"Using specified dimensions: {w}x{h}")
        elif aspect_ratio:
            w, h = self._get_dimensions(None, aspect_ratio, QWEN_SIZE_DICT)
            self._log(f"Using aspect ratio {aspect_ratio}: {w}x{h}")
        elif not use_input_dimensions:
            original_images = [images] if isinstance(images, str) else images
            w, h = self._get_dimensions(original_images, None, QWEN_SIZE_DICT, resize_mp=target_resize)
            self._log(f"Using dimensions from input image: {w}x{h}")
        else:
            self._log("Omitting dimensions - API will use input image dimensions")
        
        # Prepare images and detect if using base64 or URLs
        prepared_images, use_base64 = self._prepare_images(images, target_resize)
        
        # Get defaults from model config
        if num_inference_steps is None:
            num_inference_steps = config.get("num_inference_steps")
        
        payload = self.payload_builder.build_qwen_edit_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            width=w,  # None means omit from payload
            height=h,  # None means omit from payload
            seed=seed,
            use_base64=use_base64,
            num_inference_steps=num_inference_steps,
            **kwargs
        )
        # print(payload)
        return self._make_request(Endpoint.QWEN_EDIT, payload)
    
    def img2img_v7(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "seedream-4.0-i2i",
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> APIResponse:
        """
        Advanced image-to-image transformation (v7 API).
        
        Args:
            prompt: Transformation instructions
            images: Input images — URLs or base64 strings — can be single or list
            model_id: Model to use (seedream, gen4, flux-2-pro, wan-2.7-i2i, etc.)
            aspect_ratio: Output aspect ratio (e.g., "1:1", "16:9")
            resolution: Output resolution (e.g., "1k", "2k")
            seed: Random seed (None = random)
            **kwargs: Additional API parameters

        Returns:
            APIResponse with request status

        Notes:
            - V7 supports both public URLs and base64-encoded images
            - base64 flag is added automatically when non-URL images are detected
            - init_image can be a list
        """
        if isinstance(images, str):
            images = [images]

        # V7 requires raw base64 (no data URI prefix)
        prepared_images = []
        for img in images:
            if is_base64(img) and img.startswith("data:"):
                # Strip "data:image/xxx;base64," prefix
                img = img.split(",", 1)[1]
            prepared_images.append(img)

        payload = self.payload_builder.build_img2img_v7_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            seed=seed,
            **kwargs
        )

        # Diagnostic: log payload summary (truncate image data)
        debug_payload = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                         for k, v in payload.items()}
        print(f"[img2img_v7] payload keys: {list(payload.keys())}")
        print(f"[img2img_v7] payload (truncated): {debug_payload}")

        return self._make_request(Endpoint.IMG2IMG_V7, payload)
    
    def txt2img_v7(
        self,
        prompt: str,
        model_id: str = "grok-imagine-image-t2i",
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> APIResponse:
        """
        Text-to-image generation using V7 API.
        
        Args:
            prompt: Text description of desired image
            model_id: Model to use (e.g., "grok-imagine-image-t2i")
            aspect_ratio: Output aspect ratio (e.g., "1:1", "16:9")
            resolution: Output resolution (e.g., "1k", "2k")
            seed: Random seed (None = random)
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
        """
        payload = self.payload_builder.build_txt2img_v7_payload(
            prompt=prompt,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            seed=seed,
            **kwargs
        )
        
        return self._make_request(Endpoint.TXT2IMG_V7, payload)
    
    def generate_base(
        self,
        prompt: str,
        images: Optional[Union[str, List[str]]] = None,
        model_id: str = "flux-2-dev",
        **kwargs
    ) -> APIResponse:
        """
        Smart generation method - automatically selects endpoint based on model and inputs.
        
        Automatically determines:
        - Whether to use txt2img or img2img based on presence of images
        - Which endpoint to use based on model configuration
        - Proper init_image format (list vs string) based on model
        
        Args:
            prompt: Generation/transformation prompt
            images: Optional input images (determines txt2img vs img2img)
            model_id: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with request status
            
        Raises:
            ValueError: If model doesn't support the requested operation
        """
        config = get_model_config(model_id)
        has_images = images is not None
        
        # Determine operation type
        if has_images:
            # Check if model supports img2img
            if not model_supports_img2img(model_id):
                raise ValueError(f"Model '{model_id}' does not support img2img")
            
            # Get the appropriate endpoint for this model
            endpoint_type = config.get("endpoint_img2img", Endpoint.IMG2IMG)
            
            if endpoint_type == Endpoint.QWEN_EDIT:
                return self.qwen_edit(prompt=prompt, images=images, model_id=model_id, **kwargs)
            elif endpoint_type == Endpoint.IMG2IMG_V7:
                return self.img2img_v7(prompt=prompt, images=images, model_id=model_id, **kwargs)
            else:
                return self.img2img(prompt=prompt, images=images, model_id=model_id, **kwargs)
        else:
            # No images - txt2img
            if not model_supports_txt2img(model_id):
                raise ValueError(f"Model '{model_id}' only supports img2img (requires init_image)")
            
            # Get the appropriate endpoint for this model
            endpoint_type = config.get("endpoint_txt2img", Endpoint.TXT2IMG)
            
            if endpoint_type == Endpoint.TXT2IMG_V7:
                return self.txt2img_v7(prompt=prompt, model_id=model_id, **kwargs)
            else:
                return self.txt2img(prompt=prompt, model_id=model_id, **kwargs)
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, List[str]]] = None,
        model_id: str = "flux-2-dev",
        retries: int = 3,
        delay: float = 2.0,
        **kwargs
    ) -> APIResponse:
        """Generate with automatic retries on failure."""
        for attempt in range(1, retries + 1):
            try:
                return self.generate_base(prompt, images, model_id, **kwargs)
            except Exception as e:
                self._log(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    self._log(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self._log("All retry attempts failed.")
                    raise

    # ========== RESULT MANAGEMENT ==========
    
    def fetch_result(self, request_id: str) -> APIResponse:
        """
        Fetch result for a given request ID.
        
        Args:
            request_id: The request ID to fetch
        
        Returns:
            APIResponse with current status and output
        """
        payload = {
            "key": self.api_key,
            "request_id": request_id
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.FETCH_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        return APIResponse.from_dict(response.json())
    
    def fetch_result_(self, request_id: str) -> APIResponse:
        payload = {
            "key": self.api_key,
            "request_id": request_id
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.FETCH_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()

    def wait_for_result(
        self,
        request_id: str,
        timeout: int = 120,
        poll_interval: float = 2.0
    ) -> APIResponse:
        """
        Wait for a request to complete.
        
        Args:
            request_id: The request ID to wait for
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status checks
        
        Returns:
            APIResponse with final status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.fetch_result(request_id)
            
            if result.is_success:
                self._log(f"✓ Request {request_id} completed successfully")
                if request_id in self._pending_requests:
                    self._pending_requests.remove(request_id)
                return result
            
            if result.is_error:
                self._log(f"✗ Request {request_id} failed: {result.error_message}")
                return result
            
            self._log(f"⏳ Status: {result.status}, waiting...")
            time.sleep(poll_interval)
        
        self._log(f"⚠ Request {request_id} timed out after {timeout}s")
        return self.fetch_result(request_id)
    
    def save_result(
        self,
        response: APIResponse,
        folder: Optional[str] = None,
        show_thumbnail: bool = False,
        thumbnail_size: tuple = (200, 200)
    ) -> List[str]:
        """
        Save result images to files.
        
        Args:
            response: APIResponse with output URLs
            folder: Folder to save to (uses default if None)
            show_thumbnail: Whether to display thumbnail
            thumbnail_size: Size of thumbnail display
        
        Returns:
            List of saved file paths
        """
        save_folder = folder or self.output_folder
        saved_paths = []
        
        urls = response.output_urls # or response.future_links
        
        for i, url in enumerate(urls):
            if not url:
                continue
            
            # Generate filename from request ID
            filename = f"{response.request_id}_{i}.png"
            filepath = os.path.join(save_folder, filename)
            
            try:
                saved_path = save_image_from_url(url, filepath)
                saved_paths.append(saved_path)
                self._log(f"✓ Saved: {saved_path}")
                
                if show_thumbnail:
                    show_image_thumbnail(saved_path, thumbnail_size)
                    
            except Exception as e:
                self._log(f"✗ Failed to save {url}: {e}")
        
        # Log URLs
        self._log_urls(response)
        
        return saved_paths
    
    def _log_urls(self, response: APIResponse) -> None:
        """Log output URLs to file."""
        urls = response.output_urls or response.future_links
        if not urls:
            return
        
        log_path = os.path.join(self.output_folder, self.urls_log_file)
        
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n# {timestamp} - Request: {response.request_id}\n")
            for url in urls:
                if url:
                    f.write(f"{url}\n")
    
    # ========== BATCH OPERATIONS ==========
    
    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[str]] = None,
        model_id: str = "flux-2-dev",
        delay: float = 1.0,
        **kwargs
    ) -> List[APIResponse]:
        """
        Generate multiple images with rate limiting.
        
        Args:
            prompts: List of prompts
            images: Optional list of input images (one per prompt)
            model_id: Model to use
            delay: Seconds between requests
            **kwargs: Additional parameters
        
        Returns:
            List of APIResponse objects
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            img = images[i] if images and i < len(images) else None
            
            try:
                result = self.generate(
                    prompt=prompt,
                    images=img,
                    model_id=model_id,
                    **kwargs
                )
                results.append(result)
                
                if i < len(prompts) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                self._log(f"✗ Failed for prompt {i}: {e}")
                results.append(None)
        
        return results
    
    def fetch_pending(self) -> List[APIResponse]:
        """Fetch status for all pending requests."""
        return [self.fetch_result(rid) for rid in self._pending_requests]
    
    def wait_all_pending(
        self,
        timeout: int = 300,
        poll_interval: float = 5.0
    ) -> List[APIResponse]:
        """Wait for all pending requests to complete."""
        results = []
        start_time = time.time()
        pending = self._pending_requests.copy()
        
        while pending and (time.time() - start_time) < timeout:
            for request_id in pending[:]:
                result = self.fetch_result(request_id)
                
                if result.is_success or result.is_error:
                    results.append(result)
                    pending.remove(request_id)
                    self._pending_requests.remove(request_id)
            
            if pending:
                time.sleep(poll_interval)
        
        return results
    
    # ========== UTILITY METHODS ==========
    
    def get_pending_request_ids(self) -> List[str]:
        """Get list of pending request IDs."""
        return self._pending_requests.copy()
    
    def get_all_request_ids(self) -> List[str]:
        """Get list of all request IDs from this session."""
        return self._all_requests.copy()
    
    def clear_request_ids(self) -> None:
        """Clear the session request ID lists."""
        self._all_requests.clear()
        self._pending_requests.clear()
    
    def fetch_all_results(self) -> List[APIResponse]:
        """
        Fetch results for all request IDs from this session.
        
        Returns:
            List of APIResponse objects
        """
        return [self.fetch_result(rid) for rid in self._all_requests]
    
    def fetch_successful_results(self) -> List[APIResponse]:
        """
        Fetch only successful results from this session.
        
        Returns:
            List of APIResponse objects with status 'success'
        """
        results = []
        for rid in self._all_requests:
            result = self.fetch_result(rid)
            if result.is_success:
                results.append(result)
        return results
    
    def save_all_successful(
        self,
        folder: Optional[str] = None,
        show_thumbnail: bool = False,
        thumbnail_size: tuple = (200, 200)
    ) -> List[str]:
        """
        Fetch and save all successful images from this session.
        
        Args:
            folder: Folder to save to (uses default if None)
            show_thumbnail: Whether to display thumbnails
            thumbnail_size: Size of thumbnail display
        
        Returns:
            List of saved file paths
        """
        all_saved = []
        successful = self.fetch_successful_results()
        
        self._log(f"Found {len(successful)} successful results out of {len(self._all_requests)} requests")
        
        for result in successful:
            saved = self.save_result(result, folder, show_thumbnail, thumbnail_size)
            all_saved.extend(saved)
        
        return all_saved
    
    def load_request_ids_from_file(
        self,
        filepath: Optional[str] = None
    ) -> List[str]:
        """Load request IDs from log file."""
        log_path = filepath or os.path.join(self.output_folder, self.requests_log_file)
        request_ids = []
        
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        request_ids.append(line)
        
        return request_ids
    
    ## Utils
    def list_schedulers(self) -> List[str]:
        """Get list of available schedulers."""
        return SCHEDULER_LIST.copy()
    
    def list_models(self) -> Dict[str, Dict]:
        """Get available models with their configurations."""
        return MODEL_CONFIGS.copy()
    
    def list_txt2img_models(self) -> List[str]:
        """Get list of models that support text-to-image."""
        return [m for m in MODEL_CONFIGS if model_supports_txt2img(m)]
    
    def list_img2img_models(self) -> List[str]:
        """Get list of models that support image-to-image."""
        return [m for m in MODEL_CONFIGS if model_supports_img2img(m)]
    
    def list_v6_models(self) -> List[str]:
        """Get list of V6 API models."""
        return [m for m, c in MODEL_CONFIGS.items() if c.get("api_version") == "v6"]
    
    def list_v7_models(self) -> List[str]:
        """Get list of V7 API models."""
        return [m for m, c in MODEL_CONFIGS.items() if c.get("api_version") == "v7"]
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The model to get info for
            
        Returns:
            Dictionary with model capabilities and defaults
        """
        config = get_model_config(model_id)
        return {
            "model_id": model_id,
            "api_version": config.get("api_version", "v6"),
            "supports_txt2img": config.get("supports_txt2img", True),
            "supports_img2img": config.get("supports_img2img", True),
            "requires_scheduler_for_img2img": config.get("requires_scheduler", False),
            "init_image_as_list": config.get("init_image_as_list", True),
            "default_steps": config.get("num_inference_steps"),
            "default_strength": config.get("strength"),
        }


# ============================================================================
# RETRY DECORATOR AND HELPERS
# ============================================================================

def with_retry(
    func,
    max_retries: int = 5,
    delay: float = 1.0,
    verbose: bool = True
):
    """
    Execute a function with retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        delay: Seconds between retries
        verbose: Whether to print retry messages
    
    Returns:
        Function result or None if all retries failed
    """
    result = None
    retries = 0
    
    while result is None and retries < max_retries:
        try:
            result = func()
        except Exception as e:
            if verbose:
                print(f"Attempt {retries + 1} failed: {e}")
        
        retries += 1
        if result is None and retries < max_retries:
            time.sleep(delay)
    
    if result is None and verbose:
        print(f"Max retries ({max_retries}) reached")
    
    return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_api(api_key: Optional[str] = None, **kwargs) -> ModelsLabAPI:
    """Create and return a ModelsLabAPI instance."""
    return ModelsLabAPI(api_key=api_key, **kwargs)


# ============================================================================
# MAIN / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    # custom scripts
    from mlslab_utils import *
    from prompt_manager_page import load_yaml_file
    folder = "../images/fem/bellezze"
    handles = {
    0: "melikedhn",
    1: "rapuanomarisa",
    2: "mellaanniee",
    3: "veronicacanova",
    4: "siimonalucio",
    5: "erikaprinzi"
}
    image_bellez = get_images_paths(folder, handle=handles[0])

    folder = "G:\\Altri computer\\Horizon\\horizon_workspace\\ai-gen\\ai-art\\my-art\\my-lora\\lora_diana\\lora_train\\training_set_82"
    show_folder_images_thumbnails(folder, max_images=15, thumb_size=(10, 10))
    handle = ["03","09","14","20"]
    handle = ".jpg"
    image_files = get_images_paths(folder, handle=handle)

    # get prompts_custom.yaml
    prompts_yaml = load_yaml_file("..\prompts\prompts_custom.yaml")
    prompt_a = prompts_yaml["edit_prompts"]["realism"][1]#.keys()
    print("Promt: ", prompt_a[:100], "...\n\n")

    # Example usage
    api = ModelsLabAPI()
    test_links =[

"https://i.pinimg.com/736x/fc/d5/34/fcd5345c6e8dbbb2eb882f86726adcfa.jpg",
"https://i.pinimg.com/1200x/cf/85/9e/cf859e697fc3c839d2f64c5362f0d6c4.jpg",
"https://i.pinimg.com/1200x/48/56/4e/48564e0729e9852bd48ad049a4dafe28.jpg",
"https://i.pinimg.com/736x/a8/24/e1/a824e19da371b2f841a2f75825e61134.jpg",
"https://i.pinimg.com/736x/e0/61/57/e06157e7e2537c290910d20f6430cd67.jpg",
"https://i.pinimg.com/736x/3f/95/12/3f9512621c243e135ff821d573a127ae.jpg",
"https://i.pinimg.com/736x/38/cb/40/38cb409147733dfeb3864313df3e6d72.jpg",
"https://i.pinimg.com/736x/36/7f/88/367f8827543bad4dbd495c4aff644a91.jpg",
"https://i.pinimg.com/1200x/07/3c/6b/073c6b1e41238b508681fe1f03b3fabc.jpg",
"https://i.pinimg.com/736x/da/1d/39/da1d39b6fb2f1a801fa9e3c285d9c9ed.jpg",
"https://i.pinimg.com/736x/c8/7f/c2/c87fc228231ae59a6367fb434aebb989.jpg",
"https://i.pinimg.com/1200x/94/45/78/944578982d3a5b95968917e7d9a823a5.jpg",
"https://i.pinimg.com/736x/4c/f7/69/4cf76954fe8a1b0398ade0b1d63ebb3a.jpg",
"https://i.pinimg.com/736x/95/24/e8/9524e8d4d817776253ddf563899026f1.jpg",
"https://i.pinimg.com/1200x/14/5d/72/145d72d3092ab2f277eab0de7935d74e.jpg",
    ]
    test_link = "https://i.pinimg.com/736x/f4/17/3f/f4173ffed29e6bfe86b6c735e12bcb22.jpg"
    test_path = "..\\images\\fem\\bellezze\\angelaangelino217462786763624370684492891260608152900.jpg"
    with open("anime_prompts.json", "r", encoding="utf-8") as f:
        aniem_prompts = json.load(f)
        animew_prompts = aniem_prompts["prompts"]
        aniem_lora = 'fc-anime-lora-flux-fcanimeflux'


#%%

#%%
if __name__ == "__main__":
    
    # Text to image example
    response = api.generate(
        prompt="A beautiful Italian woman portrait, professional photography, with neon pink lipstick and nail polish, intricate details, cinematic lighting, 8k resolution",
        model_id="flux-2-dev",
        # model_id="flux-kontext-dev",
        # model_id="fluxdev",
        aspect_ratio="3:4"
    )
    print(f"Request ID: {response.request_id}")
    print(f"Status: {response.status}")
#%%
if __name__ == "__main__":
    api.fetch_successful_results()
#%%
# Image to image example
if __name__ == "__main__":
        for link in test_links[:1]:
            response = api.generate(
                # prompt="Transform the photo to anime style",
                # prompt= "Convert this photo into a DeviantArt-style digital painting, semi-realistic anime proportions, strong dramatic lighting, textured brush strokes, detailed shading, fantasy illustration vibe, keep the same character and pose",
                # prompt = "Restyle this photo into Studio Ghibli-inspired anime art, soft colors, painterly shading, gentle lighting, natural atmosphere, simplified but expressive facial features, keep the same composition and scene layout.",
                prompt= "The girl is holding a red apple with her perfect squared red nails. She has multile silver rings. Fetish style photography, high detail, sharp focus, professional lighting, 8k",
                images=[test_path],
                # images=[link],
                model_id="flux-kontext-dev",
                # model_id="fluxdev",
                # lora_model=aniem_lora,
            )
        
#%%
# Qwen edit example
if __name__ == "__main__":
        prompt = "The girl is holding a red apple with her perfect squared red nails. She has multile silver rings. Fetish style photography, high detail, sharp focus, professional lighting, 8k"
        
        for link in test_links[:1]:
            response = api.generate(
                prompt=prompt,
                images = [link],
                # images=[test_path],
                model_id="qwen-edit-2511"
            )
            time.sleep(1)
#%%
if __name__ == "__main__":
    rotate_promt = "- Show the subject from a different angle, rotating the perspective to highlight facial features. "
    r2 = "- Show the subject's face from a different angle, rotating the perspective to highlight facial features."
    prompt = rotate_promt
    prompt = prompt_a.replace("Her gaze is empty yet deeply sensual.", "")
    # prompt = prompt.replace("and vulgar.", "and vulgar. Looking at the camera with a seductive and inviting expression.")
    path = "..\images\persona\other\best.milf\56.jpg"
    path = image_files[0]
    for path in image_files[15:18]:
        response = api.generate(
                    prompt=prompt,
                    images = [path],
                    # images=[test_path],
                    model_id="qwen-edit-2511"
                )
        time.sleep(1)
#%%

#%%
if __name__ == "__main__":
    from server.image_server import ImageServer, add_image_to_server
    image_server = ImageServer(port=9999)
    vioce_sample_path = image_files[1]
    url1 = add_image_to_server(vioce_sample_path, image_server)
    
    response = api.generate(
                prompt=prompt_a,
                images = [url1],
                # images=[test_path],
                model_id="grok-imagine-image-i2i"
            )
#%%


#%%

#%%
# Wait for result and save
if __name__ == "__main__":
        result = api.wait_for_result(response.request_id)
        if result.is_success:
            api.save_result(result,folder="outputs", show_thumbnail=True)
#%%
    # Debug: print response (only in main)
    # print(response)


# %%
# test saving from api
if __name__ == "__main__":
    api.save_all_successful(folder="outputs", show_thumbnail=True)

#%%
# test saving from api
if __name__ == "__main__":
    request_ids = api.get_all_request_ids()
    print(f"Loaded {len(request_ids)} request IDs from log")
    request_id_alredy_in_folder = []
    for rid in request_ids:
        filename = f"{rid}_0.png"
        filepath = os.path.join("outputs", filename)
        if os.path.exists(filepath):
            print(f"File already exists for request {rid}, skipping: {filepath}")
            request_id_alredy_in_folder.append(rid)
            continue
        else:
            result = api.fetch_result(rid)
            print(f"Request {rid} - Status: {result.status}")
            if result.is_success:
                api.save_result(result, 
                                folder="outputs",
                                show_thumbnail=True
                                )

# %%
# ✅ CORRETTO - passa una funzione che verrà eseguita da with_retry
if __name__ == "__main__":
    response = with_retry(
        lambda: api.img2img(prompt="Transform to anime", images=[link]),
        max_retries=3
    )
# %%
