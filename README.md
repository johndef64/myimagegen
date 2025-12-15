# OpenRouter Image Generator - Streamlit App

A user-friendly Streamlit web application for generating images using OpenRouter's AI image generation models.

## Features

- 🎨 **Multiple AI Models**: Support for 6 different image generation models including FLUX, Gemini, and GPT
- 🖼️ **Reference Images**: Upload multiple reference images to guide the generation
- ⚙️ **Customizable Parameters**: Full control over all generation parameters
- 📐 **Aspect Ratios**: 10 different aspect ratio options with auto-detection from reference images
- 🎲 **Seed Control**: Use random or fixed seeds for reproducible results
- 💾 **Auto-save**: Automatic saving with metadata (prompt, model, seed, aspect ratio)
- 📜 **Generation History**: View and download previous generations
- 📥 **Easy Download**: One-click download for generated images

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit openai pillow
```

## Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Configure the API Key:**
   - Enter your OpenRouter API key in the sidebar
   - Or place it in `api_keys.json` file (see format below)

3. **Generate Images:**
   - Enter your prompt in the text area
   - (Optional) Upload reference images
   - Customize parameters in the sidebar
   - Click "Generate Image"

## Configuration

### API Key Setup

Create an `api_keys.json` file in the same directory:
```json
{
  "openrouter": "your-api-key-here"
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
vlm-tagger-prompt-gen/
├── app.py                      # Streamlit application
├── generator.py                # Original generation script
├── requirements.txt  # Dependencies
├── api_keys.json              # API keys (create this)
└── outputs/                   # Generated images (auto-created)
```

## Based On

This Streamlit app is based on the functionality from `generator.py` which provides the core image generation capabilities using OpenRouter API.

## License

This project is for personal use. Please ensure you comply with OpenRouter's terms of service and the respective model providers' usage policies.
