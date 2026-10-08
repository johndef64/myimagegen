#%%
import os
import sys
import json
import uuid
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageOps

# Add parent dir so we can import tagger
sys.path.insert(0, os.path.dirname(__file__))
from tagger import TaggerGPT, DEFAULT_SYSTEM_IMAGE_PROMPT, optimize_image

# ── Load task instructions ──────────────────────────────────────────
TASK_INSTRUCTIONS = {}
additional_tasks_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "additional_tasks.json")
if os.path.exists(additional_tasks_path):
    with open(additional_tasks_path, "r") as f:
        TASK_INSTRUCTIONS.update(json.load(f))
print(f"Available tasks: {list(TASK_INSTRUCTIONS.keys())}")

# ── CONFIG ──────────────────────────────────────────────────────────
QPG_TASK = "OUTFIT"
QPG_IMAGE_FOLDER = "../images/Outfits/"
QPG_TASK = "MAKEUP"
QPG_IMAGE_FOLDER = "../images/Makeup/"
QPG_TASK = "POSE_AND_CAMERA"
QPG_IMAGE_FOLDER = "../images\\Posers\\Feet Poses\\"
QPG_OUTPUT = f"{QPG_TASK.lower()}.json"
QPG_MODEL = "grok-4-fast"
THUMBNAIL_WIDTH = 150
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
# ────────────────────────────────────────────────────────────────────


def make_thumbnail_base64(image: Image.Image, width: int = THUMBNAIL_WIDTH) -> str:
    """Resize to `width` px (proportional height), return base64 JPEG string."""
    w, h = image.size
    new_h = int(h * (width / w))
    thumb = image.resize((width, new_h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def thumbnail_from_base64(b64: str) -> Image.Image:
    """Decode a base64 JPEG string back to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64)))


def collect_image_paths(folder: str) -> list[str]:
    """Return sorted list of image file paths in `folder`."""
    folder = Path(folder)
    paths = [
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


async def generate_caption_with_retry(
    tagger: TaggerGPT,
    instruction: str,
    image: Image.Image,
    max_retries: int = 50,
    base_delay: float = 2.0,
) -> str:
    """Generate caption via API, retrying indefinitely on failure."""
    loop = asyncio.get_event_loop()
    processed = optimize_image(image, target_size=1120)

    for attempt in range(1, max_retries + 1):
        try:
            caption = await loop.run_in_executor(
                None,
                tagger.chat_completion_prompt,
                DEFAULT_SYSTEM_IMAGE_PROMPT,
                instruction,
                processed,
            )
            if caption and caption.strip():
                return caption.strip()
            raise ValueError("Empty caption received")
        except Exception as e:
            delay = min(base_delay * (2 ** (attempt - 1)), 60)
            print(f"  ⚠ Attempt {attempt} failed: {e}  — retrying in {delay:.0f}s")
            await asyncio.sleep(delay)

    raise RuntimeError("Max retries exceeded for caption generation")


async def build_dataset(
    task: str,
    image_folder: str,
    model: str = QPG_MODEL,
    concurrency: int = 5,
) -> list[dict]:
    """Build dataset entries with thumbnails and captions (async)."""
    instruction = TASK_INSTRUCTIONS[task]
    tagger = TaggerGPT(model)
    paths = collect_image_paths(image_folder)
    print(f"Found {len(paths)} images in {image_folder}")
    print(f"Task: {task}  |  Model: {model}")
    print(f"Instruction: {instruction}\n")

    semaphore = asyncio.Semaphore(concurrency)
    dataset: list[dict] = []

    async def process_one(path: str, index: int):
        async with semaphore:
            img = Image.open(path).convert("RGB")
            thumb_b64 = make_thumbnail_base64(img)
            print(f"[{index + 1}/{len(paths)}] Captioning {Path(path).name} ...")
            caption = await generate_caption_with_retry(tagger, instruction, img)
            print(f"  ✅ {Path(path).name}: {caption[:80]}...")
            return {
                "id": str(uuid.uuid4()),
                "filename": Path(path).name,
                "path": str(Path(path).resolve()),
                "caption": caption,
                "task": task,
                "model": model,
                "thumbnail": thumb_b64,
                "created_at": datetime.now().isoformat(),
            }

    tasks = [process_one(p, i) for i, p in enumerate(paths)]
    dataset = await asyncio.gather(*tasks)
    return list(dataset)


def save_dataset(dataset: list[dict], output_path: str):
    """Save dataset to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Dataset saved to {output_path}  ({len(dataset)} entries)")


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Visualization ──────────────────────────────────────────────────
def visualize_dataset(path_or_data, columns: int = 5, max_items: int | None = None):
    """Display dataset thumbnails with captions in a grid using matplotlib."""
    import matplotlib.pyplot as plt
    import textwrap

    if isinstance(path_or_data, (str, Path)):
        data = load_dataset(str(path_or_data))
    else:
        data = path_or_data

    if max_items:
        data = data[:max_items]

    n = len(data)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 4))
    if rows == 1:
        axes = [axes] if columns == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, entry in enumerate(data):
        img = thumbnail_from_base64(entry["thumbnail"])
        axes[i].imshow(img)
        wrapped = textwrap.fill(entry["caption"], width=28)
        axes[i].set_title(wrapped, fontsize=7, pad=4)
        axes[i].axis("off")

    # Hide unused cells
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Dataset: {data[0].get('task', '?')}  ({n} images)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Main ────────────────────────────────────────────────────────────
async def main():
    dataset = await build_dataset(
        task=QPG_TASK,
        image_folder=QPG_IMAGE_FOLDER,
        model=QPG_MODEL,
    )
    save_dataset(dataset, QPG_OUTPUT)


if __name__ == "__main__":
    asyncio.run(main())

# %%
