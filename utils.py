from PIL import Image, ImageOps
import requests
from io import BytesIO
import os


IMAGES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _scan_image_folder(root: str):
    """Return list of (rel_path, abs_path) for all images under root, sorted."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)
                results.append((rel_path, abs_path))
    return results


def render_image_selector(session_key: str = "img_selector", images_root: str = None, stealth_mode: bool = False):
    """
    Streamlit widget: two cascading selectboxes (folder → subfolder),
    paged thumbnail grid, returns list[PIL.Image] or None.
    """
    import streamlit as st
    from io import BytesIO as _BytesIO
    import base64 as _b64

    PAGE_SIZE = 18
    COLS = 6
    THUMB = 110

    root = images_root or IMAGES_ROOT
    if not os.path.isdir(root):
        st.warning(f"Images folder not found: {root}")
        return None

    selected_key = f"{session_key}_selected"
    page_key = f"{session_key}_page"
    last_scan_key = f"{session_key}_last_scan"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    # --- cascading selectboxes: one per level, as deep as subfolders exist ---
    scan_root = root
    level = 0
    labels = ["Folder", "Subfolder", "Sub-subfolder"] + [f"Level {i}" for i in range(4, 20)]

    top_dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    current_dirs = top_dirs
    while current_dirs:
        options = ["(here)"] + current_dirs
        choice = st.selectbox(labels[level], options, key=f"{session_key}_lvl{level}")
        if choice == "(here)":
            break
        scan_root = os.path.join(scan_root, choice)
        level += 1
        current_dirs = sorted(d for d in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, d)))

    # reset page when browsed folder changes
    if st.session_state.get(last_scan_key) != scan_root:
        st.session_state[page_key] = 0
        st.session_state[last_scan_key] = scan_root

    # collect images (non-recursive: current dir only)
    all_images = sorted(
        os.path.join(scan_root, f)
        for f in os.listdir(scan_root)
        if os.path.isfile(os.path.join(scan_root, f))
        and os.path.splitext(f)[1].lower() in _IMG_EXTS
    )

    if not all_images:
        st.info("No images in this folder.")
    else:
        total_pages = max(1, (len(all_images) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(st.session_state[page_key], total_pages - 1)
        st.session_state[page_key] = page
        page_images = all_images[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

        n_sel = len(st.session_state[selected_key])
        sel_label = f" · {n_sel} selected" if n_sel else ""
        st.caption(f"{len(all_images)} images · page {page+1}/{total_pages}{sel_label}")

        # pagination
        pcols = st.columns([1, 4, 1])
        with pcols[0]:
            if st.button("◀", key=f"{session_key}_prev", disabled=page == 0, use_container_width=True):
                st.session_state[page_key] -= 1
                st.rerun()
        with pcols[1]:
            if total_pages > 1:
                new_page = st.select_slider(
                    " ", options=list(range(1, total_pages + 1)),
                    value=page + 1, key=f"{session_key}_slider", label_visibility="collapsed"
                )
                if new_page - 1 != page:
                    st.session_state[page_key] = new_page - 1
                    st.rerun()
        with pcols[2]:
            if st.button("▶", key=f"{session_key}_next", disabled=page >= total_pages - 1, use_container_width=True):
                st.session_state[page_key] += 1
                st.rerun()

        st.markdown(
            "<style>div[data-testid='stButton']>button{min-height:0;padding:2px 4px;font-size:13px;}</style>",
            unsafe_allow_html=True,
        )

        rows = [page_images[i:i+COLS] for i in range(0, len(page_images), COLS)]
        for row in rows:
            cols = st.columns(COLS)
            for col, abs_path in zip(cols, row):
                with col:
                    try:
                        thumb = Image.open(abs_path)
                        thumb.thumbnail((THUMB, THUMB))
                        is_selected = abs_path in st.session_state[selected_key]
                        border = "3px solid #4CAF50" if is_selected else "3px solid transparent"
                        buf = _BytesIO()
                        thumb.save(buf, format="PNG")
                        b64img = _b64.b64encode(buf.getvalue()).decode()
                        st.markdown(
                            f'<img src="data:image/png;base64,{b64img}" '
                            f'style="width:100%;border:{border};border-radius:4px;display:block;"/>',
                            unsafe_allow_html=True,
                        )
                        label = "✅ selected" if is_selected else "select"
                        if st.button(label, key=f"{session_key}_btn_{abs_path}", use_container_width=True):
                            if is_selected:
                                st.session_state[selected_key].remove(abs_path)
                            else:
                                st.session_state[selected_key].append(abs_path)
                            st.rerun()
                    except Exception:
                        pass

    # --- selection preview ---
    selected_paths = [p for p in st.session_state[selected_key] if os.path.exists(p)]
    st.session_state[selected_key] = selected_paths

    if not selected_paths:
        return None

    st.markdown(f"**{len(selected_paths)} selected:**")
    prev_cols = st.columns(min(len(selected_paths), 6))
    for i, path in enumerate(selected_paths):
        with prev_cols[i % 6]:
            try:
                if not stealth_mode:
                    st.image(Image.open(path), width=130)
            except Exception:
                pass

    if st.button("Clear selection", key=f"{session_key}_clear"):
        st.session_state[selected_key] = []
        st.rerun()

    result = []
    for path in selected_paths:
        try:
            result.append(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        except Exception:
            pass
    return result if result else None

def show_image_in_notebook(image):
    """Display a PIL Image in a Jupyter notebook."""
    from IPython.display import display
    display(image)

def load_image(image_path, resize=False, max_size=512):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    if resize:
        image = resize_image(image, max_size=max_size)
    return image

def load_image_as_list(image_paths:list, resize=False, max_size=512):
    images = []
    for path in image_paths:
        img = load_image(path, resize=resize, max_size=max_size)
        images.append(img)
    return images


def resize_image(image, max_size=512, maintain_aspect=True):
    """
    Resize image conservatively, maintaining aspect ratio and not exceeding max_size.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)
        maintain_aspect: If True, maintains aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Only resize if image is larger than max_size
        if width > max_size or height > max_size:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.copy()
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    


from PIL import Image, ImageOps

def optimize_image(image_path, output_path=None, target_size=1120, tile_size=560):
    """
    Resizes and pads an image to optimal dimensions for Llama 4 vision models.
    
    Strategy:
    1. Resizes the longest edge to 'target_size' (1120px) while maintaining aspect ratio.
    2. Pads the shorter edge to align with 'target_size' (resulting in a square 1120x1120 canvas).
    3. Uses high-quality Lanczos resampling to prevent aliasing on text/lines.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the optimized image. If None, returns PIL object.
        target_size (int): The target dimension for the square canvas (default 1120 for 2x2 grid).
        tile_size (int): The underlying tile size (default 560). 
                         (Used here for logic validation, though target_size is the driver).
    
    Returns:
        PIL.Image: The processed image object.
    """
    
    img = load_image(image_path)
    original_width, original_height = img.size
    
    # 1. Calculate new dimensions maintaining aspect ratio
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:
        # Landscape: Width becomes target_size
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Portrait/Square: Height becomes target_size
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        
    # 2. Resize using high-quality Lanczos resampling (best for downscaling)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 3. Pad to create a square canvas (1120x1120)
    # We use a neutral color (black or grey) for padding. 
    # Llama 4 generally handles black padding well.
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    
    padding = (
        delta_w // 2,        # Left
        delta_h // 2,        # Top
        delta_w - (delta_w // 2), # Right
        delta_h - (delta_h // 2)  # Bottom
    )
    
    # Add padding (fill with black: (0,0,0))
    img_padded = ImageOps.expand(img_resized, padding, fill=(0, 0, 0))
    
    # 4. Save or Return
    if output_path:
        img_padded.save(output_path, quality=95)
        print(f"Image saved to {output_path} | Size: {img_padded.size}")
    
    return img_padded

# --- Usage Example ---
# if __name__ == "__main__":
#     # Example usage for a wide chart
#     processed_img = optimize_image_for_llama4("input_chart.png", "ready_for_llama.jpg")
