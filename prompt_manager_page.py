import streamlit as st
import yaml
import os
from copy import deepcopy

def load_yaml_file(file_path="prompts.yaml"):
    """Load YAML file"""
    if os.path.exists("prompts_custom.yaml"):
        file_path = "prompts_custom.yaml"
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                # Ensure data is a dict and has required sections
                if not isinstance(data, dict):
                    st.error(f"YAML file is not in expected format (not a dictionary)")
                    return {"create_prompts": {}, "edit_prompts": {}}
                # Ensure both sections exist
                if 'create_prompts' not in data:
                    data['create_prompts'] = {}
                if 'edit_prompts' not in data:
                    data['edit_prompts'] = {}
                return data
        else:
            st.warning(f"File not found: {file_path}")
            return {"create_prompts": {}, "edit_prompts": {}}
    except Exception as e:
        st.error(f"Error loading YAML: {str(e)}")
        return {"create_prompts": {}, "edit_prompts": {}}

def save_yaml_file(data, file_path="prompts.yaml"):
    """Save data to YAML file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Error saving YAML: {str(e)}")
        return False

def get_nested_value(data, path):
    """Get value from nested dictionary using path list"""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def set_nested_value(data, path, value):
    """Set value in nested dictionary using path list"""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value

def delete_nested_key(data, path):
    """Delete key from nested dictionary using path list"""
    current = data
    for key in path[:-1]:
        if key not in current:
            return False
        current = current[key]
    if path[-1] in current:
        del current[path[-1]]
        return True
    return False

def flatten_structure(data, parent_path=None, section=""):
    """Flatten nested structure into a list with full paths"""
    if parent_path is None:
        parent_path = []
    
    items = []
    
    if not isinstance(data, dict):
        return items
    
    for key, value in data.items():
        current_path = parent_path + [key]
        if isinstance(value, list):
            # This is a prompts list
            items.append({
                'type': 'category',
                'path': current_path,
                'section': section,
                'name': key,
                'full_path': ' > '.join(current_path),
                'prompts': value,
                'level': len(current_path)
            })
        elif isinstance(value, dict):
            # Keep recursing
            items.append({
                'type': 'folder',
                'path': current_path,
                'section': section,
                'name': key,
                'full_path': ' > '.join(current_path),
                'level': len(current_path)
            })
            items.extend(flatten_structure(value, current_path, section))
    
    return items

def show_prompt_manager_page():
    """Main function to display the prompt manager page"""
    
    st.title("📝 Prompt Manager")
    st.markdown("View, edit, and create prompts in your `prompts.yaml` file")
    
    # Initialize session state
    if 'pm_yaml_data' not in st.session_state:
        st.session_state.pm_yaml_data = load_yaml_file()
    if 'pm_unsaved_changes' not in st.session_state:
        st.session_state.pm_unsaved_changes = False
    if 'pm_selected_section' not in st.session_state:
        st.session_state.pm_selected_section = "create_prompts"
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Manager Settings")
        
        # File operations
        st.subheader("💾 File Operations")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload", width="stretch", help="Reload from file (discards unsaved changes)"):
                st.session_state.pm_yaml_data = load_yaml_file()
                st.session_state.pm_unsaved_changes = False
                st.success("✅ Reloaded from file")
                st.rerun()
        
        with col2:
            save_disabled = not st.session_state.pm_unsaved_changes
            if st.button("💾 Save", width="stretch", disabled=save_disabled, 
                        type="primary" if st.session_state.pm_unsaved_changes else "secondary",
                        help="Save changes to prompts.yaml"):
                if save_yaml_file(st.session_state.pm_yaml_data):
                    st.session_state.pm_unsaved_changes = False
                    st.success("✅ Saved to prompts.yaml")
                    # Update main app's prompts data
                    if 'prompts_data' in st.session_state:
                        st.session_state.prompts_data = st.session_state.pm_yaml_data
                        from app import flatten_prompts
                        st.session_state.flattened_prompts = flatten_prompts(st.session_state.pm_yaml_data)
                    st.rerun()
        
        if st.session_state.pm_unsaved_changes:
            st.warning("⚠️ Unsaved changes")
        else:
            st.info("✅ All changes saved")
        
        st.divider()
        
        # Section selector
        st.subheader("📂 Section")
        section = st.radio(
            "Select Section",
            options=["create_prompts", "edit_prompts"],
            format_func=lambda x: "🎨 Create Prompts" if x == "create_prompts" else "✏️ Edit Prompts",
            key="section_radio"
        )
        st.session_state.pm_selected_section = section
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        create_items = flatten_structure(st.session_state.pm_yaml_data.get('create_prompts', {}), section='create_prompts')
        edit_items = flatten_structure(st.session_state.pm_yaml_data.get('edit_prompts', {}), section='edit_prompts')
        
        create_categories = [item for item in create_items if item['type'] == 'category']
        edit_categories = [item for item in edit_items if item['type'] == 'category']
        
        total_create_prompts = sum(len(cat['prompts']) for cat in create_categories)
        total_edit_prompts = sum(len(cat['prompts']) for cat in edit_categories)
        
        st.metric("Create Prompts", total_create_prompts)
        st.metric("Edit Prompts", total_edit_prompts)
        st.metric("Total Categories", len(create_categories) + len(edit_categories))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📋 View/Edit Prompts", "➕ Add New", "🗑️ Delete"])
    
    # Use the selected section from session state
    current_section = st.session_state.pm_selected_section
    
    with tab1:
        st.subheader(f"{'🎨 Create Prompts' if current_section == 'create_prompts' else '✏️ Edit Prompts'}")
        
        # Flatten current section
        section_data = st.session_state.pm_yaml_data.get(current_section, {})
        
        # Debug info
        if not section_data:
            st.error(f"⚠️ Section `{current_section}` is empty or not found in YAML data")
            st.info("Try clicking 'Reload' button to refresh from file")
            with st.expander("🔍 Debug Info"):
                st.write("Available sections:", list(st.session_state.pm_yaml_data.keys()))
                st.write("YAML data:", st.session_state.pm_yaml_data)
        
        items = flatten_structure(section_data, parent_path=None, section=current_section)
        
        if not items and section_data:
            st.warning(f"⚠️ No items found in `{current_section}`. Use the 'Add New' tab to create categories and prompts.")
            with st.expander("🔍 Debug - Raw Section Data"):
                st.write(section_data)
        elif not items:
            st.info(f"No items found in `{current_section}`. Use the 'Add New' tab to create categories and prompts.")
        else:
            # Group by category
            categories = [item for item in items if item['type'] == 'category']
            folders = [item for item in items if item['type'] == 'folder']
            
            st.write(f"**{len(categories)} categories found**")
            
            # Display folders structure (read-only view)
            if folders:
                with st.expander("📁 Folder Structure", expanded=False):
                    for folder in folders:
                        indent = "  " * (folder['level'] - 1)
                        st.text(f"{indent}📁 {folder['name']}")
            
            st.divider()
            
            # Display and edit each category
            for idx, category in enumerate(categories):
                with st.expander(f"📂 {category['full_path']} ({len(category['prompts'])} prompts)", expanded=False):
                    st.caption(f"**Path:** `{' > '.join(category['path'])}`")
                    
                    prompts = category['prompts']
                    
                    # Display each prompt with edit capability
                    for prompt_idx, prompt in enumerate(prompts):
                        # Skip comment lines
                        if isinstance(prompt, str) and prompt.strip().startswith('#'):
                            st.caption(f"💬 Comment: `{prompt}`")
                            continue
                        
                        st.markdown(f"**Prompt {prompt_idx + 1}:**")
                        
                        # Editable text area for each prompt
                        edited_prompt = st.text_area(
                            f"Edit prompt",
                            value=prompt if isinstance(prompt, str) else str(prompt),
                            height=100,
                            key=f"edit_{section}_{idx}_{prompt_idx}",
                            label_visibility="collapsed"
                        )
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            if edited_prompt != prompt:
                                if st.button(f"💾 Save Changes", key=f"save_{section}_{idx}_{prompt_idx}", type="primary"):
                                    # Update the prompt
                                    current_prompts = get_nested_value(st.session_state.pm_yaml_data[section], category['path'])
                                    current_prompts[prompt_idx] = edited_prompt
                                    set_nested_value(st.session_state.pm_yaml_data[section], category['path'], current_prompts)
                                    st.session_state.pm_unsaved_changes = True
                                    st.success("✅ Prompt updated")
                                    st.rerun()
                        
                        with col2:
                            # Copy to clipboard
                            st.button(f"📋 Copy", key=f"copy_{section}_{idx}_{prompt_idx}", 
                                     help="Click to copy prompt", width="stretch")
                        
                        with col3:
                            # Delete prompt
                            if st.button(f"🗑️ Delete", key=f"del_{section}_{idx}_{prompt_idx}", 
                                        help="Delete this prompt", width="stretch"):
                                current_prompts = get_nested_value(st.session_state.pm_yaml_data[section], category['path'])
                                current_prompts.pop(prompt_idx)
                                set_nested_value(st.session_state.pm_yaml_data[section], category['path'], current_prompts)
                                st.session_state.pm_unsaved_changes = True
                                st.success("✅ Prompt deleted")
                                st.rerun()
                        
                        st.divider()
                    
                    # Add new prompt to this category
                    st.markdown("**➕ Add New Prompt to this Category:**")
                    new_prompt = st.text_area(
                        "New prompt",
                        height=100,
                        key=f"new_prompt_{section}_{idx}",
                        label_visibility="collapsed",
                        placeholder="Enter new prompt text..."
                    )
                    
                    if st.button(f"➕ Add Prompt", key=f"add_prompt_{section}_{idx}", type="primary"):
                        if new_prompt.strip():
                            current_prompts = get_nested_value(st.session_state.pm_yaml_data[section], category['path'])
                            current_prompts.append(new_prompt.strip())
                            set_nested_value(st.session_state.pm_yaml_data[section], category['path'], current_prompts)
                            st.session_state.pm_unsaved_changes = True
                            st.success("✅ Prompt added")
                            st.rerun()
                        else:
                            st.warning("⚠️ Prompt cannot be empty")
    
    with tab2:
        st.subheader("➕ Add New Category or Folder")
        
        add_type = st.radio(
            "What would you like to add?",
            options=["Category with Prompts", "Folder (Container)"],
            horizontal=True
        )
        
        st.divider()
        
        if add_type == "Category with Prompts":
            st.markdown("### Create a new category with prompts")
            st.caption("A category is the final level that contains actual prompt text.")
            
            # Path input
            st.markdown("**Category Path:**")
            st.caption("Use '>' to separate levels, e.g., `girls_and_nails > closeups > nails`")
            
            path_input = st.text_input(
                "Path",
                placeholder="e.g., girls_and_nails > closeups > nails",
                key="new_category_path",
                label_visibility="collapsed"
            )
            
            # Prompts input
            st.markdown("**Prompts:**")
            st.caption("Enter prompts, one per line")
            prompts_input = st.text_area(
                "Prompts",
                height=200,
                placeholder="Enter prompts, one per line...",
                key="new_category_prompts",
                label_visibility="collapsed"
            )
            
            if st.button("➕ Create Category", type="primary", key="create_category_btn"):
                if not path_input.strip():
                    st.error("❌ Path cannot be empty")
                elif not prompts_input.strip():
                    st.error("❌ Please enter at least one prompt")
                else:
                    # Parse path
                    path_parts = [p.strip() for p in path_input.split('>')]
                    
                    # Parse prompts
                    prompts = [p.strip() for p in prompts_input.split('\n') if p.strip()]
                    
                    # Check if category already exists
                    existing = get_nested_value(st.session_state.pm_yaml_data[section], path_parts)
                    if existing is not None:
                        st.error(f"❌ Category already exists at path: `{' > '.join(path_parts)}`")
                    else:
                        # Create the category
                        set_nested_value(st.session_state.pm_yaml_data[section], path_parts, prompts)
                        st.session_state.pm_unsaved_changes = True
                        st.success(f"✅ Created category: `{' > '.join(path_parts)}` with {len(prompts)} prompts")
                        st.rerun()
        
        else:  # Folder
            st.markdown("### Create a new folder")
            st.caption("A folder is a container that can hold other folders or categories.")
            
            # Path input
            st.markdown("**Folder Path:**")
            st.caption("Use '>' to separate levels, e.g., `girls_and_nails > closeups`")
            
            folder_path_input = st.text_input(
                "Path",
                placeholder="e.g., girls_and_nails > closeups",
                key="new_folder_path",
                label_visibility="collapsed"
            )
            
            if st.button("➕ Create Folder", type="primary", key="create_folder_btn"):
                if not folder_path_input.strip():
                    st.error("❌ Path cannot be empty")
                else:
                    # Parse path
                    path_parts = [p.strip() for p in folder_path_input.split('>')]
                    
                    # Check if folder already exists
                    existing = get_nested_value(st.session_state.pm_yaml_data[section], path_parts)
                    if existing is not None:
                        st.error(f"❌ Item already exists at path: `{' > '.join(path_parts)}`")
                    else:
                        # Create the folder (empty dict)
                        set_nested_value(st.session_state.pm_yaml_data[section], path_parts, {})
                        st.session_state.pm_unsaved_changes = True
                        st.success(f"✅ Created folder: `{' > '.join(path_parts)}`")
                        st.rerun()
    
    with tab3:
        st.subheader("🗑️ Delete Categories or Folders")
        st.warning("⚠️ **Warning:** Deleting a folder will delete all its contents!")
        
        # Flatten current section
        section_data = st.session_state.pm_yaml_data.get(section, {})
        items = flatten_structure(section_data, section=section)
        
        if not items:
            st.info(f"No items to delete in `{section}`.")
        else:
            st.markdown("**Select item to delete:**")
            
            # Create list of all items (folders and categories)
            all_items = sorted(items, key=lambda x: x['full_path'])
            
            for idx, item in enumerate(all_items):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    icon = "📂" if item['type'] == 'category' else "📁"
                    prompt_count = f"({len(item['prompts'])} prompts)" if item['type'] == 'category' else "(folder)"
                    st.text(f"{icon} {item['full_path']} {prompt_count}")
                
                with col2:
                    st.caption(f"Level {item['level']}")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_item_{section}_{idx}", type="secondary"):
                        # Confirm deletion
                        if delete_nested_key(st.session_state.pm_yaml_data[section], item['path']):
                            st.session_state.pm_unsaved_changes = True
                            st.success(f"✅ Deleted: `{item['full_path']}`")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete item")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>Prompt Manager • Manage your prompts.yaml file</p>
    </div>
    """, unsafe_allow_html=True)
