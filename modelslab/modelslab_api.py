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

# Model configurations with default parameters
MODEL_CONFIGS = {
    "qwen": {
        "num_inference_steps": 8,
        "strength": 0.5,
        "endpoint": Endpoint.IMG2IMG,
    },
    "qwen-edit": {
        "num_inference_steps": 8,
        "endpoint": Endpoint.QWEN_EDIT,
    },
    "qwen-edit-2511": {
        "num_inference_steps": 8,
        "endpoint": Endpoint.QWEN_EDIT,
    },
    "flux-kontext-dev": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "endpoint": Endpoint.IMG2IMG,
    },
    "flux-2-dev": {
        "num_inference_steps": 28,
        "strength": 0.7,
        "endpoint": Endpoint.IMG2IMG,
    },
    "z-image-base": {
        "num_inference_steps": 20,
        "strength": 0.7,
        "endpoint": Endpoint.IMG2IMG,
    },
    "z-image-turbo": {
        "num_inference_steps": 8,
        "strength": 0.7,
        "endpoint": Endpoint.IMG2IMG,
    },
    "seedream-4.0-i2i": {
        "endpoint": Endpoint.IMG2IMG_V7,
    },
    "gen4_image_turbo": {
        "endpoint": Endpoint.IMG2IMG_V7,
    },
    "flux-2-pro": {
        "endpoint": Endpoint.IMG2IMG_V7,
    },
    "nano-banana": {
        "endpoint": Endpoint.IMG2IMG_V7,
    },
}


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
    return data.startswith("data:image") or data.startswith("/9j/")


def get_images_paths(folder: str, handle: str = "") -> List[str]:
    """
    Get list of image file paths from a folder.
    
    Args:
        folder: Folder path to search
        handle: Optional filter string to match in filename
    
    Returns:
        List of matching image file paths
    """
    import glob
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
    filepath: str
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
        img_data = decode_base64_to_image(response.text)
    else:
        img_data = response.content
    
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
    Modular payload builder for different API endpoints.
    
    Builds request payloads based on endpoint type, allowing easy
    customization and extension of parameters.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def build_base_payload(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """Build common base payload parameters."""
        payload = {
            "key": self.api_key,
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
        }
        # Add any additional kwargs
        payload.update(kwargs)
        return payload
    
    def build_txt2img_payload(
        self,
        prompt: str,
        model_id: str = "flux-2-dev",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        scheduler: str = "DPMSolverMultistepScheduler",
        enhance_prompt: str = "no",
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for text2img endpoint."""
        payload = self.build_base_payload(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed
        )
        payload.update({
            "model_id": model_id,
            "num_inference_steps": str(num_inference_steps),
            "guidance_scale": str(guidance_scale),
            "scheduler": scheduler,
            "enhance_prompt": enhance_prompt,
            "base64": "yes",
            "temp": "yes",
        })
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        
        if lora_model:
            payload["lora_model"] = lora_model
        if lora_strength is not None:
            payload["lora_strength"] = str(lora_strength)
        
        payload.update(kwargs)
        return payload
    
    def build_img2img_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "flux-kontext-dev",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        scheduler: str = "DPMSolverMultistepScheduler",
        strength: float = 0.7,
        enhance_prompt: str = "no",
        use_base64: str = "yes",
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for img2img endpoint."""
        payload = self.build_base_payload(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed
        )
        
        # Handle image inputs
        init_image = images[0] if images else None
        init_image_2 = images[1] if len(images) > 1 else None
        
        payload.update({
            "model_id": model_id,
            "init_image": init_image,
            "init_image_2": init_image_2,
            "num_inference_steps": str(num_inference_steps),
            "guidance_scale": str(guidance_scale),
            "scheduler": scheduler,
            "strength": str(strength),
            "enhance_prompt": enhance_prompt,
            "base64": use_base64,
            "temp": "yes",
        })
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        
        if lora_model:
            payload["lora_model"] = lora_model
        if lora_strength is not None:
            payload["lora_strength"] = str(lora_strength)
        
        payload.update(kwargs)
        return payload
    
    def build_qwen_edit_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "qwen-edit",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        use_base64: str = "yes",
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for qwen_edit endpoint."""
        payload = self.build_base_payload(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed
        )
        
        payload.update({
            "init_image": images,
            "base64": use_base64,
        })
        
        # Only add model_id if not default qwen-edit
        if model_id and model_id != "qwen-edit":
            payload["model_id"] = model_id
        
        payload.update(kwargs)
        return payload
    
    def build_img2img_v7_payload(
        self,
        prompt: str,
        images: List[str],
        model_id: str = "seedream-4.0-i2i",
        aspect_ratio: str = "1:1",
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for v7 img2img endpoint."""
        valid_ratios = ["1:1", "4:3", "9:16", "16:9", "3:2", "2:3", "21:9", "9:21"]
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "1:1"
        
        payload = {
            "key": self.api_key,
            "init_image": images,
            "prompt": prompt,
            "model_id": model_id,
            "aspect-ratio": aspect_ratio,
            "seed": seed,
        }
        
        payload.update(kwargs)
        return payload


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
        guidance_scale: float = 7.5,
        scheduler: str = "DPMSolverMultistepScheduler",
        seed: int = -1,
        enhance_prompt: str = "no",
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
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
            num_inference_steps: Generation steps (higher = quality, slower)
            guidance_scale: CFG scale (default 7.5, typically 1.0-20.0)
            scheduler: Scheduler algorithm (default "DPMSolverMultistepScheduler")
            seed: Random seed (-1 for random)
            enhance_prompt: Whether to enhance prompt ("yes"/"no")
            lora_model: LoRA model ID to use (e.g., "flux-krea-realism-lora-v1-0")
            lora_strength: LoRA strength (0.0-1.0)
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
        """
        # Determine dimensions
        if width and height:
            w, h = width, height
        else:
            w, h = self._get_dimensions(None, aspect_ratio)
        
        # Get default steps from model config
        if num_inference_steps is None:
            config = MODEL_CONFIGS.get(model_id, {})
            num_inference_steps = config.get("num_inference_steps", 20)
        
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
            **kwargs
        )
        
        return self._make_request(Endpoint.TXT2IMG, payload)
    
    def img2img(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "flux-kontext-dev",
        aspect_ratio: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        scheduler: str = "DPMSolverMultistepScheduler",
        strength: Optional[float] = None,
        seed: int = -1,
        resize_mp: Optional[float] = None,
        lora_model: Optional[str] = None,
        lora_strength: Optional[float] = None,
        **kwargs
    ) -> APIResponse:
        """
        Transform existing image based on prompt.
        
        Args:
            prompt: Text description of desired transformation
            images: Input image(s) - paths or URLs
            model_id: Model to use
            aspect_ratio: Output aspect ratio
            width: Output width
            height: Output height
            negative_prompt: What to avoid
            num_inference_steps: Generation steps
            guidance_scale: CFG scale (default 7.5, typically 1.0-20.0)
            scheduler: Scheduler algorithm (default "DPMSolverMultistepScheduler")
            strength: How much to transform (0.0-1.0)
            seed: Random seed
            resize_mp: Resize input to this megapixel size
            lora_model: LoRA model ID to use
            lora_strength: LoRA strength (0.0-1.0)
            **kwargs: Additional API parameters
        

        Returns:
            APIResponse with request status
        """
        # Determine resize target
        target_resize_mp = resize_mp or self.default_resize_mp
        
        # Determine dimensions FIRST (before converting to base64)
        # This way we can read dimensions from local files
        if width and height:
            w, h = width, height
            print(f"Using specified dimensions: {w}x{h}")
        elif aspect_ratio:
            w, h = self._get_dimensions(None, aspect_ratio)
            print(f"Using aspect ratio {aspect_ratio}: {w}x{h}")
        else:
            original_images = [images] if isinstance(images, str) else images
            w, h = self._get_dimensions(original_images, aspect_ratio, resize_mp=target_resize_mp)
            print(f"Using dimensions from input image (resized to {target_resize_mp}MP): {w}x{h}")
        
        # Prepare images and detect if using base64 or URLs
        prepared_images, use_base64 = self._prepare_images(images, target_resize_mp)
        
        # Get defaults from model config
        config = MODEL_CONFIGS.get(model_id, {})
        if num_inference_steps is None:
            num_inference_steps = config.get("num_inference_steps", 28)
        if strength is None:
            strength = config.get("strength", 0.7)
        
        payload = self.payload_builder.build_img2img_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            width=w,
            height=h,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler=scheduler,
            strength=strength,
            use_base64=use_base64,
            lora_model=lora_model,
            lora_strength=lora_strength,
            **kwargs
        )
        
        return self._make_request(Endpoint.IMG2IMG, payload)
    
    def qwen_edit(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "qwen-edit",
        aspect_ratio: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = -1,
        resize_mp: Optional[float] = None,
        **kwargs
    ) -> APIResponse:
        """
        Edit image using Qwen model.
        
        Args:
            prompt: Edit instructions
            images: Input image(s) - up to 4 images supported
            model_id: "qwen-edit" or "qwen-edit-2511"
            aspect_ratio: Output aspect ratio
            width: Output width
            height: Output height
            seed: Random seed
            resize_mp: Resize input to this megapixel size
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
        """
        # Determine resize target (Qwen supports higher resolution)
        target_resize = resize_mp or 1.7
        
        # Determine dimensions FIRST (before converting to base64)
        if width and height:
            w, h = width, height
        else:
            original_images = [images] if isinstance(images, str) else images
            w, h = self._get_dimensions(original_images, aspect_ratio, QWEN_SIZE_DICT, resize_mp=target_resize)
        
        # Prepare images and detect if using base64 or URLs
        prepared_images, use_base64 = self._prepare_images(images, target_resize)
        
        payload = self.payload_builder.build_qwen_edit_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            width=w,
            height=h,
            seed=seed,
            use_base64=use_base64,
            **kwargs
        )
        
        return self._make_request(Endpoint.QWEN_EDIT, payload)
    
    def img2img_v7(
        self,
        prompt: str,
        images: Union[str, List[str]],
        model_id: str = "seedream-4.0-i2i",
        aspect_ratio: str = "1:1",
        seed: int = -1,
        **kwargs
    ) -> APIResponse:
        """
        Advanced image-to-image transformation (v7 API).
        
        Args:
            prompt: Transformation instructions
            images: Input image URLs (v7 requires URLs)
            model_id: Model to use (seedream, gen4, etc.)
            aspect_ratio: Output aspect ratio
            seed: Random seed
            **kwargs: Additional API parameters
        
        Returns:
            APIResponse with request status
        """
        # V7 requires URLs - prepare images accordingly
        if isinstance(images, str):
            images = [images]
        
        # Check if we need to handle local files
        prepared_images = []
        for img in images:
            if not is_url(img):
                raise ValueError(f"V7 endpoint requires URLs. Got local path: {img}")
            prepared_images.append(img)
        
        payload = self.payload_builder.build_img2img_v7_payload(
            prompt=prompt,
            images=prepared_images,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
            seed=seed,
            **kwargs
        )
        
        return self._make_request(Endpoint.IMG2IMG_V7, payload)
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, List[str]]] = None,
        model_id: str = "flux-2-dev",
        **kwargs
    ) -> APIResponse:
        """
        Smart generation method - automatically selects endpoint based on model and inputs.
        
        Args:
            prompt: Generation/transformation prompt
            images: Optional input images
            model_id: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with request status
        """
        config = MODEL_CONFIGS.get(model_id, {})
        endpoint = config.get("endpoint", Endpoint.IMG2IMG if images else Endpoint.TXT2IMG)
        
        if endpoint == Endpoint.TXT2IMG:
            return self.txt2img(prompt=prompt, model_id=model_id, **kwargs)
        elif endpoint == Endpoint.QWEN_EDIT:
            return self.qwen_edit(prompt=prompt, images=images, model_id=model_id, **kwargs)
        elif endpoint == Endpoint.IMG2IMG_V7:
            return self.img2img_v7(prompt=prompt, images=images, model_id=model_id, **kwargs)
        else:
            return self.img2img(prompt=prompt, images=images, model_id=model_id, **kwargs)
    
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
        
        urls = response.output_urls or response.future_links
        
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
    
    def list_schedulers(self) -> List[str]:
        """Get list of available schedulers."""
        return SCHEDULER_LIST.copy()
    
    def list_models(self) -> Dict[str, Dict]:
        """Get available models with their configurations."""
        return MODEL_CONFIGS.copy()


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
