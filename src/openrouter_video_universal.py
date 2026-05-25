"""
OpenRouter Universal Video Generation Client
=============================================

Esempio completo e modulare per generare video usando QUALSIASI modello
disponibile su OpenRouter (Kling, Veo, Wan, Seedance/ByteDance, Hailuo,
Sora, Grok Imagine, ...) tramite l'endpoint asincrono /api/v1/videos.

Documentazione di riferimento:
- https://openrouter.ai/docs/guides/overview/multimodal/video-generation
- https://openrouter.ai/docs/api/api-reference/video-generation/create-videos
- https://openrouter.ai/docs/api/api-reference/video-generation/list-videos-models

PARAMETRI STANDARD NORMALIZZATI (validi cross-modello)
------------------------------------------------------
- model              : str   (obbligatorio)   es. "kwaivgi/kling-v3.0-pro"
- prompt             : str   (obbligatorio)
- duration           : int   secondi (es. 5, 8, 10, 15)
- resolution         : str   "480p" | "720p" | "1080p" | "1K" | "2K" | "4K"
- aspect_ratio       : str   "16:9" | "9:16" | "1:1" | "4:3" | "3:4" |
                             "3:2"  | "2:3"  | "21:9" | "9:21"
- size               : str   "WIDTHxHEIGHT" (es. "1280x720") — alternativa
                             a resolution + aspect_ratio
- generate_audio     : bool  default True per i modelli che supportano audio
- seed               : int   seed deterministico (non garantito da tutti)
- frame_images       : list  image-to-video (first_frame / last_frame)
- input_references   : list  reference-to-video (style/content guidance)
- callback_url       : str   webhook HTTPS (opzionale)
- provider           : dict  passthrough parameters provider-specifici

PASSTHROUGH PARAMETERS provider-specifici
-----------------------------------------
Esempi noti:
- Veo (google-vertex):  personGeneration, negativePrompt, ...
- Wan / Seedance / Kling: parametri esposti nel campo
  `allowed_passthrough_parameters` di /api/v1/videos/models

Per scoprire dinamicamente cosa supporta un modello:
    client.list_models()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests


# ---------------------------------------------------------------------------
# CATALOGO MODELLI (snapshot Maggio 2026 - puoi sempre verificare via API)
# ---------------------------------------------------------------------------
# Slug ufficiali OpenRouter. Per la lista live e i parametri supportati di
# ogni modello, usare client.list_models().
VIDEO_MODELS: dict[str, str] = {
    # Kuaishou Kling
    "kling-v3-pro":      "kwaivgi/kling-v3.0-pro",
    "kling-v3-std":      "kwaivgi/kling-v3.0-std",
    "kling-o1":          "kwaivgi/kling-video-o1",
    # Google Veo
    "veo-3.1":           "google/veo-3.1",
    "veo-3.1-fast":      "google/veo-3.1-fast",
    "veo-3.1-lite":      "google/veo-3.1-lite",
    # ByteDance Seedance
    "seedance-2":        "bytedance/seedance-2.0",
    "seedance-2-fast":   "bytedance/seedance-2.0-fast",
    # Alibaba Wan
    "wan-2.7":           "alibaba/wan-2.7",
    "wan-2.6":           "alibaba/wan-2.6",
    # MiniMax
    "hailuo-2.3":        "minimax/hailuo-2.3",
    # OpenAI / xAI (verifica disponibilità via list_models)
    "sora-2-pro":        "openai/sora-2-pro",
    "grok-imagine":      "xai/grok-imagine",
}


# ---------------------------------------------------------------------------
# Helper per costruire frame_images / input_references
# ---------------------------------------------------------------------------
def make_frame_image(url: str, frame_type: str = "first_frame") -> dict:
    """
    Costruisce un'entry frame_images per image-to-video.
    frame_type: "first_frame" oppure "last_frame".
    """
    assert frame_type in ("first_frame", "last_frame")
    return {
        "type": "image_url",
        "image_url": {"url": url},
        "frame_type": frame_type,
    }


def make_input_reference(url: str) -> dict:
    """Costruisce un'entry input_references per reference-to-video."""
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


# ---------------------------------------------------------------------------
# Request dataclass — tutti i parametri possibili
# ---------------------------------------------------------------------------
@dataclass
class VideoRequest:
    model: str
    prompt: str

    # Output shape
    duration: int | None = None
    resolution: str | None = None        # 480p | 720p | 1080p | 1K | 2K | 4K
    aspect_ratio: str | None = None      # 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3, 21:9, 9:21
    size: str | None = None              # "1280x720" — alternativa a resolution+aspect_ratio

    # Audio & determinism
    generate_audio: bool | None = None
    seed: int | None = None

    # Image conditioning
    frame_images: list[dict] = field(default_factory=list)
    input_references: list[dict] = field(default_factory=list)

    # Webhook
    callback_url: str | None = None

    # Provider passthrough — es:
    # {"options": {"google-vertex": {"parameters": {"personGeneration": "allow"}}}}
    provider: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Esporta un payload JSON-serializable senza campi None/vuoti."""
        payload: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        optionals = {
            "duration": self.duration,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "size": self.size,
            "generate_audio": self.generate_audio,
            "seed": self.seed,
            "callback_url": self.callback_url,
            "provider": self.provider,
        }
        for k, v in optionals.items():
            if v is not None:
                payload[k] = v
        if self.frame_images:
            payload["frame_images"] = self.frame_images
        if self.input_references:
            payload["input_references"] = self.input_references
        return payload


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class OpenRouterVideoClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_referer: str | None = None,    # opzionale, per attribution
        x_title: str | None = None,         # opzionale, per attribution
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY mancante (env o param).")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            self._headers["HTTP-Referer"] = http_referer
        if x_title:
            self._headers["X-Title"] = x_title

    # --- Discovery ----------------------------------------------------------
    def list_models(self) -> list[dict]:
        """
        Ritorna la lista live dei modelli video con i loro parametri:
        supported_resolutions, supported_aspect_ratios, supported_durations,
        supported_frame_images, allowed_passthrough_parameters, pricing_skus,
        generate_audio, ecc.
        """
        r = requests.get(
            f"{self.BASE_URL}/videos/models",
            headers=self._headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("data", [])

    def get_model_info(self, model_id: str) -> dict | None:
        for m in self.list_models():
            if m.get("id") == model_id or m.get("canonical_slug") == model_id:
                return m
        return None

    # --- Generation ---------------------------------------------------------
    def submit(self, req: VideoRequest) -> dict:
        """POST /api/v1/videos — ritorna {id, polling_url, status, ...}."""
        r = requests.post(
            f"{self.BASE_URL}/videos",
            headers=self._headers,
            data=json.dumps(req.to_payload()),
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(f"Submit fallito {r.status_code}: {r.text}")
        return r.json()

    def poll(self, polling_url: str) -> dict:
        """GET sul polling_url ritornato dal submit."""
        # polling_url può essere assoluto o relativo (/api/v1/videos/...)
        if polling_url.startswith("/"):
            polling_url = f"https://openrouter.ai{polling_url}"
        r = requests.get(polling_url, headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait_for_completion(
        self,
        polling_url: str,
        *,
        interval: float = 10.0,
        max_wait: float = 900.0,  # 15 min
        verbose: bool = True,
    ) -> dict:
        """Polling con backoff lineare fino a status terminale."""
        start = time.time()
        terminal = {"completed", "failed", "cancelled", "expired"}
        last_status = None
        while True:
            status = self.poll(polling_url)
            s = status.get("status")
            if verbose and s != last_status:
                print(f"[{int(time.time() - start)}s] status: {s}")
                last_status = s
            if s in terminal:
                return status
            if time.time() - start > max_wait:
                raise TimeoutError(f"Job non terminato entro {max_wait}s")
            time.sleep(interval)

    def download(self, content_url: str, out_path: str) -> str:
        """Scarica il video da unsigned_urls[i] -> file locale."""
        if content_url.startswith("/"):
            content_url = f"https://openrouter.ai{content_url}"
        with requests.get(
            content_url, headers=self._headers, stream=True, timeout=self.timeout
        ) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return out_path

    # --- High-level convenience --------------------------------------------
    def generate(
        self,
        req: VideoRequest,
        *,
        out_path: str | None = None,
        poll_interval: float = 10.0,
        max_wait: float = 900.0,
        verbose: bool = True,
    ) -> dict:
        """Submit + poll + (optionally) download. Ritorna lo stato finale."""
        sub = self.submit(req)
        if verbose:
            print(f"Job: {sub.get('id')}  status: {sub.get('status')}")
        result = self.wait_for_completion(
            sub["polling_url"],
            interval=poll_interval,
            max_wait=max_wait,
            verbose=verbose,
        )
        if result.get("status") == "completed" and out_path:
            urls = result.get("unsigned_urls") or []
            if urls:
                self.download(urls[0], out_path)
                result["saved_to"] = out_path
                if verbose:
                    print(f"Salvato in: {out_path}")
        return result


# ---------------------------------------------------------------------------
# ESEMPI D'USO
# ---------------------------------------------------------------------------
def example_text_to_video_kling():
    """Text-to-video con Kling v3.0 Pro, audio incluso."""
    client = OpenRouterVideoClient()
    req = VideoRequest(
        model=VIDEO_MODELS["kling-v3-pro"],
        prompt="A serene mountain landscape at sunset with clouds drifting by",
        duration=10,
        resolution="1080p",
        aspect_ratio="16:9",
        generate_audio=True,
        seed=42,
    )
    return client.generate(req, out_path="kling_out.mp4")


def example_text_to_video_veo_with_passthrough():
    """Veo 3.1 con passthrough provider-specifico (personGeneration, negativePrompt)."""
    client = OpenRouterVideoClient()
    req = VideoRequest(
        model=VIDEO_MODELS["veo-3.1"],
        prompt="A time-lapse of a flower blooming in a sunlit meadow",
        duration=8,
        resolution="1080p",
        aspect_ratio="16:9",
        generate_audio=True,
        provider={
            "options": {
                "google-vertex": {
                    "parameters": {
                        "personGeneration": "allow",
                        "negativePrompt": "blurry, low quality, distorted",
                    }
                }
            }
        },
    )
    return client.generate(req, out_path="veo_out.mp4")


def example_image_to_video_wan():
    """Image-to-video con Wan 2.7 — first_frame + last_frame."""
    client = OpenRouterVideoClient()
    req = VideoRequest(
        model=VIDEO_MODELS["wan-2.7"],
        prompt="The character walks slowly into the forest as fog rolls in",
        resolution="1080p",
        aspect_ratio="16:9",
        duration=5,
        frame_images=[
            make_frame_image("https://example.com/first.png", "first_frame"),
            make_frame_image("https://example.com/last.png", "last_frame"),
        ],
    )
    return client.generate(req, out_path="wan_out.mp4")


def example_reference_to_video_seedance():
    """Reference-to-video con Seedance 2.0 (preserva stile/character)."""
    client = OpenRouterVideoClient()
    req = VideoRequest(
        model=VIDEO_MODELS["seedance-2"],
        prompt="The same character drinking coffee in a Parisian cafe",
        size="1280x720",          # alternativa a resolution+aspect_ratio
        duration=8,
        input_references=[
            make_input_reference("https://example.com/character_ref.png"),
        ],
    )
    return client.generate(req, out_path="seedance_out.mp4")


def example_discover_model_capabilities():
    """Stampa cosa supporta ogni modello (utile prima di submit)."""
    client = OpenRouterVideoClient()
    for m in client.list_models():
        print(f"\n=== {m.get('id')} ({m.get('name')}) ===")
        print(f"  Durations:        {m.get('supported_durations')}")
        print(f"  Resolutions:      {m.get('supported_resolutions')}")
        print(f"  Aspect ratios:    {m.get('supported_aspect_ratios')}")
        print(f"  Sizes:            {m.get('supported_sizes')}")
        print(f"  Frame images:     {m.get('supported_frame_images')}")
        print(f"  Audio gen:        {m.get('generate_audio')}")
        print(f"  Passthrough:      {m.get('allowed_passthrough_parameters')}")
        print(f"  Pricing SKUs:     {m.get('pricing_skus')}")


if __name__ == "__main__":
    # Imposta OPENROUTER_API_KEY in env, poi de-commenta uno degli esempi:
    #
    # example_discover_model_capabilities()
    # example_text_to_video_kling()
    # example_text_to_video_veo_with_passthrough()
    # example_image_to_video_wan()
    # example_reference_to_video_seedance()
    pass
