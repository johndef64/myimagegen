import streamlit as st
import os
from PIL import Image
from pathlib import Path
import json

def get_image_metadata(image_path):
    """Extract metadata from PNG image"""
    try:
        img = Image.open(image_path)
        metadata = {}
        if hasattr(img, 'text'):
            metadata = img.text
        return metadata
    except:
        return {}

def get_all_images_recursive(root_path):
    """Get all images from root path and all subfolders"""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
    images = []
    
    try:
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_path)
                    images.append({
                        'filename': file,
                        'full_path': full_path,
                        'rel_path': rel_path,
                        'folder': os.path.dirname(rel_path) if os.path.dirname(rel_path) else "Root",
                        'size': os.path.getsize(full_path)
                    })
    except Exception as e:
        st.error(f"Error scanning directory: {str(e)}")
    
    return images

def get_subfolders(root_path):
    """Get all subfolders in the root path"""
    try:
        subfolders = ["All Folders", "Root Only"]
        for root, dirs, files in os.walk(root_path):
            for d in dirs:
                rel_path = os.path.relpath(os.path.join(root, d), root_path)
                subfolders.append(rel_path)
        return sorted(subfolders)
    except:
        return ["All Folders", "Root Only"]

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def show_image_viewer_page():
    """Main function to display the image viewer page"""
    
    st.title("🖼️ Image Viewer")
    st.markdown("Browse and view images from your outputs folder")
    
    # Initialize session state for viewer
    if 'viewer_root_path' not in st.session_state:
        st.session_state.viewer_root_path = "outputs"
    if 'viewer_selected_folder' not in st.session_state:
        st.session_state.viewer_selected_folder = "All Folders"
    if 'viewer_grid_columns' not in st.session_state:
        st.session_state.viewer_grid_columns = 3
    if 'viewer_sort_by' not in st.session_state:
        st.session_state.viewer_sort_by = "Newest First"
    if 'viewer_show_comparisons' not in st.session_state:
        st.session_state.viewer_show_comparisons = False
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Viewer Settings")
        
        # Root path configuration
        st.subheader("📁 Root Path")
        root_path_input = st.text_input(
            "Output Root Path",
            value=st.session_state.viewer_root_path,
            help="Enter the root path to scan for images"
        )
        
        if root_path_input != st.session_state.viewer_root_path:
            st.session_state.viewer_root_path = root_path_input
            st.session_state.viewer_selected_folder = "All Folders"
        
        # Check if path exists
        if not os.path.exists(st.session_state.viewer_root_path):
            st.error(f"❌ Path does not exist: `{st.session_state.viewer_root_path}`")
            st.info("💡 Please enter a valid directory path")
            return
        
        st.success(f"✅ Scanning: `{st.session_state.viewer_root_path}`")
        
        st.divider()
        
        # Folder filter
        st.subheader("📂 Filter by Folder")
        subfolders = get_subfolders(st.session_state.viewer_root_path)
        selected_folder = st.selectbox(
            "Select Folder",
            options=subfolders,
            index=subfolders.index(st.session_state.viewer_selected_folder) if st.session_state.viewer_selected_folder in subfolders else 0,
            help="Choose a specific folder or view all images"
        )
        st.session_state.viewer_selected_folder = selected_folder
        
        st.divider()
        
        # Display settings
        st.subheader("🎨 Display Settings")
        grid_columns = st.slider(
            "Grid Columns",
            min_value=1,
            max_value=6,
            value=st.session_state.viewer_grid_columns,
            help="Number of columns in the image grid"
        )
        st.session_state.viewer_grid_columns = grid_columns
        
        # Sort options
        sort_by = st.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First", "Filename A-Z", "Filename Z-A", "Largest First", "Smallest First"],
            index=["Newest First", "Oldest First", "Filename A-Z", "Filename Z-A", "Largest First", "Smallest First"].index(st.session_state.viewer_sort_by),
            help="Sort images by different criteria"
        )
        st.session_state.viewer_sort_by = sort_by
        
        st.divider()
        
        # Show comparisons toggle
        show_comparisons = st.checkbox(
            "Show comparisons",
            value=st.session_state.viewer_show_comparisons,
            help="Show images ending with _comparison.jpg"
        )
        st.session_state.viewer_show_comparisons = show_comparisons
        
        # Show metadata toggle
        show_metadata = st.checkbox(
            "Show Image Metadata",
            value=True,
            help="Display metadata information when viewing images"
        )
    
    # Get all images
    all_images = get_all_images_recursive(st.session_state.viewer_root_path)
    
    if not all_images:
        st.warning(f"⚠️ No images found in `{st.session_state.viewer_root_path}`")
        st.info("💡 Generate some images first or check your path settings")
        return
    
    # Filter images by selected folder
    if selected_folder == "All Folders":
        filtered_images = all_images
    elif selected_folder == "Root Only":
        filtered_images = [img for img in all_images if img['folder'] == "Root"]
    else:
        filtered_images = [img for img in all_images if img['rel_path'].startswith(selected_folder)]
    
    # Filter out comparison images if checkbox is not checked
    if not st.session_state.viewer_show_comparisons:
        filtered_images = [img for img in filtered_images if not img['filename'].endswith('_comparison.jpg')]
    
    # Sort images
    if sort_by == "Newest First":
        filtered_images.sort(key=lambda x: os.path.getmtime(x['full_path']), reverse=True)
    elif sort_by == "Oldest First":
        filtered_images.sort(key=lambda x: os.path.getmtime(x['full_path']))
    elif sort_by == "Filename A-Z":
        filtered_images.sort(key=lambda x: x['filename'].lower())
    elif sort_by == "Filename Z-A":
        filtered_images.sort(key=lambda x: x['filename'].lower(), reverse=True)
    elif sort_by == "Largest First":
        filtered_images.sort(key=lambda x: x['size'], reverse=True)
    elif sort_by == "Smallest First":
        filtered_images.sort(key=lambda x: x['size'])
    
    # Display stats
    st.subheader("📊 Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(all_images))
    with col2:
        st.metric("Filtered Images", len(filtered_images))
    with col3:
        total_size = sum(img['size'] for img in filtered_images)
        st.metric("Total Size", format_file_size(total_size))
    with col4:
        unique_folders = len(set(img['folder'] for img in all_images))
        st.metric("Folders", unique_folders)
    
    st.divider()
    
    # Display images in grid
    if filtered_images:
        st.subheader(f"🖼️ Images ({len(filtered_images)})")
        
        # Create grid
        for i in range(0, len(filtered_images), grid_columns):
            cols = st.columns(grid_columns)
            for j in range(grid_columns):
                idx = i + j
                if idx < len(filtered_images):
                    img_data = filtered_images[idx]
                    with cols[j]:
                        try:
                            # Load and display image
                            img = Image.open(img_data['full_path'])
                            st.image(img, use_container_width=True)
                            
                            # Display info in expander
                            with st.expander("ℹ️ Info", expanded=False):
                                st.caption(f"**File:** {img_data['filename']}")
                                st.caption(f"**Folder:** {img_data['folder']}")
                                st.caption(f"**Size:** {format_file_size(img_data['size'])}")
                                st.caption(f"**Dimensions:** {img.size[0]} × {img.size[1]}")
                                
                                # Show metadata if enabled
                                if show_metadata:
                                    metadata = get_image_metadata(img_data['full_path'])
                                    if metadata:
                                        st.caption("**Metadata:**")
                                        for key, value in metadata.items():
                                            # Truncate long values
                                            display_value = value[:100] + "..." if len(value) > 100 else value
                                            st.caption(f"- {key}: {display_value}")
                                    else:
                                        st.caption("_No metadata found_")
                                
                                # Download button
                                with open(img_data['full_path'], 'rb') as f:
                                    st.download_button(
                                        label="📥 Download",
                                        data=f.read(),
                                        file_name=img_data['filename'],
                                        mime="image/png",
                                        key=f"download_{idx}",
                                        use_container_width=True
                                    )
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
    else:
        st.info(f"No images found in the selected folder: `{selected_folder}`")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>Image Viewer • Browse your generated images</p>
    </div>
    """, unsafe_allow_html=True)
