#%%
from openai import OpenAI
from datetime import datetime
import random
from io import BytesIO
import base64
import os
from openai import OpenAI
from datetime import datetime
import random
from utils import load_image

OPENROUTER_IMAGE_MODELS = {
    "flux.2-pro":"black-forest-labs/flux.2-pro",
    "flux.2-flex":"black-forest-labs/flux.2-flex",
    "gemini-2.5-flash-image":"google/gemini-2.5-flash-image",
    "gemini-3-pro-image-preview":"google/gemini-3-pro-image-preview",
    "gpt-5-image-mini":"openai/gpt-5-image-mini",
    "gpt-5-image":"openai/gpt-5-image"}

client_dict = {
    "openai": "",
    "grok": "https://api.x.ai/v1",
    "claude": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1/",
    "deepseek": "https://api.deepseek.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

def load_api_keys(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
    
api_dict = load_api_keys("api_keys.json")

def get_client(client_name):
    api_key = api_dict.get(client_name, "")
    base_url = client_dict.get(client_name, "")
    print(f"Using {client_name} with base URL: {base_url}")
    print(f"API Key: {api_key[:4]}****")
    return OpenAI(api_key=api_key, base_url=base_url)

image_client = get_client("openrouter")


def encode_image_to_base64(image):
            """ BASE 64 IMAGE ECODER FROM PILLOW IMAGE 
        
        """
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return image_data

def get_image_aspect_ratio(image):
        """
        get str image aspect ratio like 1:1, 3:4, etc. approssimamando al rapporto più vicino.
        Supported aspect ratios in Openrouter:
        1:1 → 1024×1024 (default)
        2:3 → 832×1248
        3:2 → 1248×832
        3:4 → 864×1184
        4:3 → 1184×864
        4:5 → 896×1152
        5:4 → 1152×896
        9:16 → 768×1344
        16:9 → 1344×768
        21:9 → 1536×672
        """
        width, height = image.size
        ratio = width / height
        aspect_ratios = {
            "1:1": 1.0,
            "2:3": 2/3,
            "3:2": 3/2,
            "3:4": 3/4,
            "4:3": 4/3,
            "4:5": 4/5,
            "5:4": 5/4,
            "9:16": 9/16,
            "16:9": 16/9,
            "21:9": 21/9
        }
        closest_ratio = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
        return closest_ratio[0]

def generate_image(prompt, 
                   MODEL_NAME=OPENROUTER_IMAGE_MODELS["flux.2-pro"],
                   aspect_ratio="1:1",
                   show_image=True, 
                   save_image=False, 
                   seed=None, 
                   images=None,
                   use_image_aspect_ratio=False
                   ):

    if not seed:
        seed = random.randint(1, 1000000)

    user_message = {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
    

    if images:
        images = [images] if not isinstance(images, list) else images
        if isinstance(images[0], str):
            print("Loading images from paths...")
            images = load_image_as_list(images, resize=True, max_size=1024)

        if use_image_aspect_ratio:
            aspect_ratio = get_image_aspect_ratio(images[0])
            print(f"Using image aspect ratio: {aspect_ratio}")

        for img in images:
            print("Processing image for prompt...")
            image_data = encode_image_to_base64(img)
            image_append = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            }
            user_message["content"].append(image_append)

    

    response_full = image_client.chat.completions.create(
      model=MODEL_NAME,
      messages=[
            #   {
            #     "role": "user",
            #     "content": prompt
            #   }
                user_message
            ],
      seed=seed,
      extra_body={"modalities": ["image", "text"],
                  "image_config": {"aspect_ratio": aspect_ratio}
                  },
  
    )
    # The generated image will be in the assistant message
    response = response_full.choices[0].message
    if response.images:
        print(f" {len(response.images)} images generated with model {MODEL_NAME}.")
        for image in response.images:
            base64_data = image['image_url']['url']  # Base64 data URL
            print(f"Generated image: {base64_data[:50]}...")

            # display the image
            from IPython.display import display
            from PIL import Image
            from io import BytesIO
            import base64
            header, encoded = base64_data.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_data))
            if show_image:
                display(image)
            if save_image:
                save_image_with_metadata(image, prompt, MODEL_NAME)
    return image_data

def save_image_with_metadata(image, prompt, model_name):
    from PIL import PngImagePlugin
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Model", model_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # use only first 20 chars of prompt for filename
    prompt_short = prompt[:20].replace(" ", "_").replace("\n", "_")
    model_name_short = model_name.split("/")[-1]
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{prompt_short}_{model_name_short}_{timestamp}.png")

    image.save(filename, pnginfo=metadata)

# get prompt from ../prompts.test.yaml
# load

def load_test_prompts(file_path="../prompts.test.yaml"):
    with open(file_path, 'r') as f:
        import yaml
        prompts_data = yaml.safe_load(f)
        scope_test_prompt(prompts_data)
        return prompts_data["prompts"]

def scope_test_prompt(prompts_data):
    print("Prompts data structure:")
    for key, value in prompts_data['prompts'].items():
        print(f"- {key}:")
        for subkey, subvalue in value.items():
            print(f"  - {subkey}: {len(subvalue)} entries")

def load_image_as_list(image_paths:list, resize=False, max_size=512):
    images = []
    for path in image_paths:
        img = load_image(path, resize=resize, max_size=max_size)
        images.append(img)
    return images


if __name__ == "__main__":
    images_paths_1 = [
        "images/photo_edit/_mellaanniee_1687632424_3132410267929820618_1279912309.jpg",
        "images/photo_edit/_mellaanniee_1687632424_3132410267946458537_1279912309.jpg",
        "images/photo_edit/_mellaanniee_1687632424_3132410267946478433_1279912309.jpg"
    ]

    images_paths_2 = [
        "images/photo_edit/siimonalucio_1761576954_3752701948949431710_33134809472.jpg",
        "images/photo_edit/siimonalucio_1761576954_3752701948949451610_33134809472.jpg",
        "images/photo_edit/siimonalucio_1761576954_3752701948949452937_33134809472.jpg",
        "images/photo_edit/siimonalucio_1761576954_3752701948957842386_33134809472.jpg"
    ]

    images_paths_3 = [
        "images/photo_edit/_mellaanniee_1683494387_3097697897480415280_1279912309.jpg",
        "images/photo_edit/_mellaanniee_1683494387_3097697897471966088_1279912309.jpg",
    ]
    MAX_SIZE = 560
    images_paths = images_paths_1
    images = load_image_as_list(images_paths, resize=True, max_size=MAX_SIZE)

    prompt = "The girls is sensually licking a melting wet popsicle, she has squared black nails and perfect hands. She is looking at the viewer"
    # prompt = "The girls is sensually sucking a dildo, drooling wet saliva, she has squared black nails and perfect hands. She is looking at the viewer"
    #   Her wet tongue is out licking the popsicle.

    model_name = "flux.2-pro"
    # model_name = "gemini-3-pro-image-preview"

    seed = 12345
    # display(images[1])
    if False:
        generate_image(prompt, images=images[1], 
                    MODEL_NAME=OPENROUTER_IMAGE_MODELS[model_name],
                    show_image=False, 
                    save_image=True, 
                    #    aspect_ratio="3:4",
                    use_image_aspect_ratio=True,
                    seed=seed)
        
    if False:
        generate_image(prompt, images=images, 

                    MODEL_NAME=OPENROUTER_IMAGE_MODELS["flux.2-pro"],
                    show_image=True, save_image=True, 
                    #    aspect_ratio="3:4",
                    use_image_aspect_ratio=True,
                    seed=seed)

#%%
# ref image selector 
# get all tyoes of images paths from  a folder
def get_image_paths_from_folder(folder_path):
    import os
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp")
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
    filenames = [os.path.basename(path) for path in image_paths]
    return image_paths, filenames

def filer_image_paths_by_keyword(image_paths, keyword):
    filtered_paths = [path for path in image_paths if keyword.lower() in os.path.basename(path).lower()]
    return filtered_paths

images_folder = "G:\\Altri computer\\Horizon\\horizon_workspace\\ai-gen\\my-refs\\"
images_folder = images_folder+"insta-pinterest\\insta\\bellezze"


# show images from images_paths
import matplotlib.pyplot as plt
def show_images(image_paths):
    images = load_image_as_list(image_paths, resize=True, max_size=256)
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

_, filenames = get_image_paths_from_folder(images_folder)
image_paths = filer_image_paths_by_keyword(_, "sevda_gorusurem   ".replace(" ", ""))
for i, filename in enumerate(filenames):
    print(f"{i}: {filename}")

show_images(image_paths)
#%%
if True:
    prompt = "A girl is sensually licking a small and wet popsicle, she has squared black nails and perfect hands. She is looking at the viewer"
    prompt = """A girl is showing her hands with squared nails painted in black, very beautiful and shiny nails, showing the nails from different angles. Intagram style, high quality photo, soft lighting, detailed skin texture, 8k resolution, shallow depth of field, bokeh background.""" #High detailed hands and nails. 
    prompt = """A girl is holding a popsicle showing her squared black nails, showing the shiny nails from different angles. Intagram style, fetish style, high quality photo, soft lighting, detailed skin texture, 8k resolution, shallow depth of field""" 
    prompt = """A girl is touching her lips and chest showing her squared black nails, showing the shiny nails from different angles. Intagram style, high quality photo, soft lighting, detailed skin texture, 8k resolution, shallow depth of field""" 

    model_name = "flux.2-pro"
    model_name = "gemini-3-pro-image-preview"
    # model_name = "gpt-5-image-mini"
    generate_image(prompt, images=image_paths[2], 

                MODEL_NAME=OPENROUTER_IMAGE_MODELS[model_name],
                show_image=False, 
                save_image=True, 
                #    aspect_ratio="3:4",
                use_image_aspect_ratio=True,
                seed=seed)
image_paths[0]

#%%

with open("../prompts.test.yaml", 'r') as f:
    import yaml
    prompts_data = yaml.safe_load(f)
    test_prompt = prompts_data['prompts']["girls_and_nails"]["posing"][0]
    print("Test prompt:", test_prompt)
#%%

if __name__ == "__main__":
    for model_name in OPENROUTER_IMAGE_MODELS.values():
        print(f"Generating image with model: {model_name}")
        generate_image(test_prompt, 
                       MODEL_NAME=model_name, 
                       seed=12345,
                       show_image=False, 
                       save_image=True)
        print("Image generated and saved.")
#%%
if __name__ == "__main__":
    model_name = "google/gemini-3-pro-image-preview"
    print(f"Generating image with model: {model_name}")
    generate_image(test_prompt, 
                    MODEL_NAME=model_name, 
                    seed=12345,
                    show_image=False, 
                    save_image=True)
    print("Image generated and saved.")
