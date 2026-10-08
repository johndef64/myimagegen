# OpenRouter Image Generator - Streamlit App

A comprehensive Streamlit application for AI image generation, management, and analysis using OpenRouter's API.

## Features

### 🎨 Image Generation Page
- **Multiple AI Models**: 6 models including FLUX.2-Pro, FLUX.2-Flex, Gemini 2.5/3, GPT-5
- **Reference Images**: Multi-image upload with auto aspect ratio detection
- **Smart Prompting**: Quick prompts from built-in YAML template library (create & edit prompts)
- **10 Aspect Ratios**: From 1:1 square to 21:9 ultrawide with auto-detection
- **Seed Control**: Random or fixed seeds for reproducible results
- **Auto-save**: Images saved with embedded metadata (prompt, model, seed, aspect ratio)

### 🖼️ Image Viewer Page
- **Recursive Browsing**: View all images from outputs folder and subfolders
- **Smart Filtering**: Filter by folder, sort by date/name/size
- **Grid View**: Adjustable 1-6 column layout
- **Metadata Display**: View generation parameters from saved images
- **Quick Navigation**: Easy subfolder navigation and image preview

### 📝 Prompt Manager Page
- **YAML Editor**: Visual interface for `prompts.yaml` and `prompts_custom.yaml`
- **Organized Library**: Create/Edit prompt categories with nested structure
- **Live Editing**: Add, edit, delete prompts and categories
- **Bulk Operations**: Manage entire prompt collections
- **Auto-save**: Changes sync with generation page

### 🏷️ VLM Tagger Page
- **Multi-Provider Support**: Groq, OpenRouter, and xAI (Grok) integration
- **Smart Tagging**: Generate tags, detailed tags, prompts from images
- **Caption Generation**: Detailed and highly-detailed captions
- **Custom Focus**: Predefined focus modes (cuteness, artistic, fantasy) via `tagger_focus.json`
- **Batch Processing**: Analyze multiple images with customizable settings
- **Image Optimization**: Automatic resize and padding for optimal VLM processing

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Required packages: `streamlit`, `openai`, `pillow`, `pyyaml`

## Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Configure API Keys:**
   - Enter API keys in the sidebar or create `api_keys.json` (see format below)
   - OpenRouter required for image generation
   - Groq/xAI/OpenRouter optional for VLM tagging

3. **Use the App:**
   - **Generate**: Enter prompt, select model, optional reference images, generate
   - **Viewer**: Browse and filter generated images from outputs folder
   - **Prompts**: Create/edit prompt templates in YAML library
   - **Tagger**: Analyze images with VLM models to generate tags/captions/prompts

## Configuration

### API Key Setup

Create an `api_keys.json` file in the same directory:
```json
{
  "openrouter": "your-api-key-here",
  "groq": "your-groq-api-key-here",
  "xai": "your-xai-api-key-here"
}
```

Or enter it directly in the sidebar when running the app.

### Available Models

- **flux.2-pro**: High-quality image generation
- **flux.2-flex**: Flexible image generation
- **gemini-2.5-flash-image**: Fast Gemini image generation
- **gemini-3-pro-image-preview**: Advanced Gemini model
- **gpt-5-image-mini**: Compact GPT image model
- **gpt-5-image**: Full GPT image model
- **and more ...**

### Aspect Ratios

- 1:1 → 1024×1024 (Square)
- 2:3 → 832×1248 (Portrait)
- 3:2 → 1248×832 (Landscape)
- 3:4 → 864×1184 (Portrait)
- 4:3 → 1184×864 (Landscape)
- 4:5 → 896×1152 (Portrait)
- 5:4 → 1152×896 (Landscape)
- 9:16 → 768×1344 (Mobile Portrait)
- 16:9 → 1344×768 (Widescreen)
- 21:9 → 1536×672 (Ultrawide)

## Parameters

### Generation Parameters
- **Model**: Choose from 6 available AI models
- **Aspect Ratio**: Select target aspect ratio or auto-detect from reference images
- **Seed**: Use random seed or set a fixed seed for reproducibility
- **Max Image Size**: Control the maximum dimension for reference images (256-2048px)

### Reference Images
- Upload multiple images (PNG, JPG, JPEG, WEBP, BMP)
- Automatically resized to max dimension before sending to API
- Optional auto-detection of aspect ratio from first reference image

### Saving Options
- **Auto-save**: Automatically save generated images to `outputs/` folder
- **Metadata**: Images saved with embedded metadata (prompt, model, seed, aspect ratio)
- **Naming**: Files named with prompt snippet, model, and timestamp

## Output

Generated images are saved to the `outputs/` directory with the following naming convention:
```
{prompt_snippet}_{model_name}_{timestamp}.png
```

Example: `A_fluffy_cat_yawning_flux.2-pro_20251205_143022.png`

## Tips

1. **For Consistent Results**: Use a fixed seed instead of random
2. **With Reference Images**: Enable "Auto-detect aspect ratio" to match your reference
3. **Better Quality**: Use flux.2-pro or gemini-3-pro-image-preview models
4. **Faster Generation**: Use flux.2-flex or gemini-2.5-flash-image models
5. **Complex Prompts**: Be detailed and specific in your descriptions

## Troubleshooting

### "No API key" error
- Make sure you've entered your OpenRouter API key in the sidebar
- Or create an `api_keys.json` file with your key

### "Failed to generate image" error
- Check your API key is valid
- Verify you have credits in your OpenRouter account
- Try a different model

### Images not saving
- Check that the `outputs/` directory can be created
- Verify write permissions in the application directory

## Project Structure

```
myimagegen/
├── app.py                      # Main Streamlit app with multi-page navigation
├── api_generator.py            # Core image generation logic
├── image_viewer_page.py        # Image browsing and viewing interface
├── prompt_manager_page.py      # YAML prompt editor
├── tagger_page.py              # VLM image analysis tools
├── prompts.yaml                # Default prompt templates
├── prompts_custom.yaml         # User-created custom prompts
├── tagger_focus.json           # Custom VLM focus configurations
├── api_keys.json               # API keys (create this)
├── requirements.txt            # Dependencies
└── outputs/                    # Generated images (auto-created)
```

## License

This project is for personal use. Please ensure you comply with OpenRouter's terms of service and the respective model providers' usage policies.
