import os
import json
import glob
import streamlit as st
from pyperclip import copy


def render_dataset_browser():
    """Render the dataset browser section in Streamlit."""
    database_dir = os.path.join(os.path.dirname(__file__), "database")
    dataset_files = sorted(glob.glob(os.path.join(database_dir, "*.json")))
    dataset_files = [f for f in dataset_files if os.path.basename(f) != "api_keys.json"]

    if not dataset_files:
        return

    st.divider()
    st.subheader("📂 Dataset Browser")

    for ds_file_idx, ds_file_path in enumerate(dataset_files):
        ds_name = os.path.splitext(os.path.basename(ds_file_path))[0]

        with st.expander(f"🗂️ Browse: {ds_name}", expanded=False):
            with open(ds_file_path, "r", encoding="utf-8") as df:
                dataset_items = json.load(df)

            if not dataset_items:
                st.info("This dataset is empty.")
            else:
                st.markdown(f"**{len(dataset_items)} items**")

                # Pagination
                items_per_page = 12
                total_pages = max(1, (len(dataset_items) + items_per_page - 1) // items_per_page)
                ds_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=f"ds_page_{ds_file_idx}")
                page_start = (ds_page - 1) * items_per_page
                page_items = dataset_items[page_start:page_start + items_per_page]

                # Display thumbnails in a grid
                cols = st.columns(5)
                for idx, item in enumerate(page_items):
                    with cols[idx % 5]:
                        caption_text = item.get("caption", "No caption")
                        thumb = item.get("thumbnail", "")

                        if thumb:
                            try:
                                st.image(
                                    f"data:image/jpeg;base64,{thumb}",
                                    use_container_width=True
                                )
                            except Exception:
                                st.write("🖼️ *Thumbnail unavailable*")
                        else:
                            st.write("🖼️ *No thumbnail*")

                        short_caption = caption_text[:80] + "..." if len(caption_text) > 80 else caption_text
                        st.caption(short_caption)

                        if st.button("📋 Copy Caption", key=f"ds_copy_{ds_file_idx}_{ds_page}_{idx}"):
                            try:
                                copy(caption_text)
                                st.success("Caption copied!")
                            except Exception:
                                pass
                            st.session_state["ds_selected_caption"] = caption_text

    # Show selected caption outside the expanders
    # if "ds_selected_caption" in st.session_state and st.session_state["ds_selected_caption"]:
    #     st.markdown("**📌 Selected Caption:**")
    #     st.code(st.session_state["ds_selected_caption"], language=None)
