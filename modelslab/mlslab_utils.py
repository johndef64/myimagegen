#%%
from matplotlib.pylab import seed
import requests
import json
import base64
import time
import os
from image_params import resize_image_to_megapixels


api_key_path = "../api_keys.json"
with open(api_key_path, "r", encoding="utf-8") as f:
    api_keys = json.load(f)
    api_key = api_keys.get("modelslab")
    print("Using ModelsLab API Key:", api_key[:4]+"..."+api_key[-4:])

os.environ["MODELSTAB_API_KEY"] = api_key


def encode_image_to_base64_prefix(image_path):
    """Converte un'immagine locale in base64"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        # Aggiungi il prefisso data URI
        return f"data:image/jpeg;base64,{encoded}"

def encode_image_to_base64(image_path, resize=None):
    """Converte un'immagine locale in stringa base64"""
    if resize:
        # ridimensiona l'immagine prima di convertire in base64
        new_width, new_height, resized_img = resize_image_to_megapixels(image_path, target_mp=resize)
        from io import BytesIO
        buffered = BytesIO()
        resized_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Saving URLs and results

## resposnse update 

def update_results_file(result, urls_file="qwen_edit_results.json"):
    with open(urls_file, "a", encoding="utf-8") as f:
        # prima aggiungi la riga della data
        from datetime import datetime
        f.write(f"\n# Edited on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Optional: save only successful results
        # if res.get("status") == "success":
        for img_url in result.get("future_links", ["miss"]):
            if img_url:                 # skip empty strings
                request_id = result.get("id", "unknown")
                f.write(img_url.strip() + f"\n- Request ID: {request_id}\n")

def update_requests_file(result, urls_file="requests_list.txt"):
    if not os.path.exists(urls_file):
        with open(urls_file, "w", encoding="utf-8") as f:
            f.write("# Qwen Edit Requests\n")
    # All image URLs must be saved in a text file, appending line-by-line
    """Aggiorna il file di testo con gli URL delle immagini modificate"""
    with open(urls_file, "a", encoding="utf-8") as f:
        # prima aggiungi la riga della data
        from datetime import datetime
        f.write(f"\n# Requests on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        request_id = result.get("id", "miss")
        f.write(str(request_id))

def update_urls_file_all(results, urls_file="edited_image_urls.txt"):
    # All image URLs must be saved in a text file, appending line-by-line
    """Aggiorna il file di testo con gli URL delle immagini modificate"""
    with open(urls_file, "a", encoding="utf-8") as f:
        # prima aggiungi la riga della data
        from datetime import datetime
        f.write(f"\n# Edited on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for res in results:
            # Optional: save only successful results
            # if res.get("status") == "success":
            for img_url in res.get("future_links", ["miss"]):
                if img_url:                 # skip empty strings
                    request_id = res.get("id", "unknown")
                    f.write(img_url.strip() + f"\n- Request ID: {request_id}\n")

## Saving results

def save_base64_images_in_results(results, folder="edited_images"):
    """Salva le immagini in base64 in file locali"""
    import os
    root_folder = os.getcwd()
    if not os.path.exists(os.path.join(root_folder, folder)):
        os.makedirs(os.path.join(root_folder, folder))
    timestamp = int(time.time())
    
    for i, res in enumerate(results):
        if True: #res.get("status") == "success":
            for j, img_base64 in enumerate(res.get("output_base64", [])):
                # Rimuovi il prefisso data URI se presente
                if img_base64.startswith("data:image"):
                    img_base64 = img_base64.split(",")[1]
                
                img_data = base64.b64decode(img_base64)
                file_path = os.path.join(root_folder, folder, f"edited_image_{i}_{j}_{timestamp}.png")
                with open(file_path, "wb") as img_file:
                    img_file.write(img_data)
                print(f"✓ Salvata immagine base64 in: {file_path}")

def save_image_from_base64url(img_url, folder, show_thumb=False):
    root_folder = os.getcwd()
    os.makedirs(os.path.join(root_folder, folder), exist_ok=True)
    """Salva un'immagine in base64 da un URL in un file locale
    url example:https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/temp/08a459fd-538e-41a8-9c14-dc009296b37f-0.base64
    """
    import requests
    if img_url:

        url_id = img_url.split("/")[-1].split(".")[0]

        response = requests.get(img_url)
        response.raise_for_status()
        base64_data = response.text
        # Rimuovi il prefisso data URI se presente
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(",")[1]
        
        img_data = base64.b64decode(base64_data)

        file_path = os.path.join(root_folder, folder, f"edited_image_{url_id}.png")

        with open(file_path, "wb") as img_file:
            img_file.write(img_data)
        print(f"✓ Salvata immagine base64 da URL in: {file_path}")
        if show_thumb:
            from IPython.display import display, Image
            display(Image(filename=file_path, width=100, height=100))
    else:
        print("URL immagine vuoto, impossibile salvare.")

def show_image_thumbnail(img_url, size=(100, 100)):
    """Mostra un'anteprima di un'immagine in base64 da un URL
    url example:https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/temp/08a459fd-538e-41a8-9c14-dc009296b37f-0.base64
    """
    import requests
    if img_url:
        is_url = img_url.startswith("http:") or img_url.startswith("https:")
        if not is_url:
            response = requests.get(img_url)
            response.raise_for_status()
            base64_data = response.text
            # Rimuovi il prefisso data URI se presente
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]
            
            img_data = base64.b64decode(base64_data)

            from IPython.display import display, Image
            from io import BytesIO
            display(Image(data=img_data, width=size[0], height=size[1]))
        else:   
            from IPython.display import display, Image
            display(Image(url=img_url, width=size[0], height=size[1]))
    else:
        print("URL immagine vuoto, impossibile mostrare.")


def fetch_image_by_requestid(api_key, request_id):
    """
    Recupera un'immagine usando l'API Community (non Enterprise)
    
    Args:
        api_key: str - La tua API key standard di ModelsLab
        request_id: str - L'ID della richiesta
    
    Returns:
        dict: Response con status e URL dell'immagine
    """
    url = "https://modelslab.com/api/v6/images/fetch"
    
    payload = json.dumps({
        "key": api_key,
        "request_id": request_id
    })
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

def save_image_from_requestid_base64(img_url, request_id, folder="fetched_images"):
    """Salva un'immagine in base64 da un URL in un file locale"""
    import os
    import requests
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if img_url:
        response = requests.get(img_url)
        response.raise_for_status()
        base64_data = response.text
        # Rimuovi il prefisso data URI se presente
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(",")[1]
        
        img_data = base64.b64decode(base64_data)

        file_path = os.path.join(folder, f"fetched_image_{request_id}.png")

        with open(file_path, "wb") as img_file:
            img_file.write(img_data)
        print(f"✓ Salvata immagine base64 da URL in: {file_path}")
    else:
        print("URL immagine vuoto, impossibile salvare.")

def get_requestsid_from_file(file_name="requests_list.txt"):
    """Legge gli ID delle richieste da un file di testo"""
    request_ids = []
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Assuming request IDs start with '168'
                    request_ids.append(line)
    return request_ids


# image managment

def show_folder_images_thumbnails(folder_path, max_images=None, thumb_size=(100, 100)):
    """small 100x100 thumbnails of all images in a folder with filename below"""
    import os
    from IPython.display import display, Image
    if not max_images:
        max_images = len(os.listdir(folder_path))

    for filename in os.listdir(folder_path)[:max_images]:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            display(Image(filename=os.path.join(folder_path, filename), width=thumb_size[0], height=thumb_size[1]))
            print(filename)


# file manaegr
   
def get_images_paths(folder, handle=""):
    import glob
    image_files = glob.glob(f"{folder}/*{handle}*")
    return image_files



############ MOdels FUNCTIONS ############

# SizeImageDict = {
# # square
# "1:1": (1024, 1024),
# # vertical
# "3:4": (768, 1024),
# "1:2": (512, 1024),
# "9:16": (576, 1024),
# #landscape
# "4:3": (1024, 768),
# "2:1": (1024, 512),
# "16:9": (1024, 576),   
# }

SizeImageDict = {
    # square
    "1:1": (1024, 1024),  # 1,048,576 px ✓
    
    # vertical
    "3:4": (886, 1182),   # 1,047,252 px (~1MP)
    "1:2": (724, 1448),   # 1,048,352 px (~1MP)
    "9:16": (764, 1358),  # 1,037,512 px (~1MP)
    
    # landscape
    "4:3": (1182, 886),   # 1,047,252 px (~1MP)
    "2:1": (1448, 724),   # 1,048,352 px (~1MP)
    "16:9": (1358, 764),  # 1,037,512 px (~1MP)
}


SchedulerList = [
    "DDPMScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",   ###
    "DPMSolverMultistepScheduler",
    "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "DPMSolverSinglestepScheduler",  ###
    "KDPM2AncestralDiscreteScheduler",
    "UniPCMultistepScheduler",  ###
    "DDIMInverseScheduler",
    "DEISMultistepScheduler",
    "IPNDMScheduler",
    "KarrasVeScheduler",
    "ScoreSdeVeScheduler",
    "LCMScheduler"
]

def show_scheduler_options():
    url = "https://modelslab.com/api/v1/enterprise/schedulers_list"
    payload = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    schedulers = response.json()
    print(schedulers)



"""Modifica l'immagine usando Qwen Edit"""

# Rest API example request payload

from typing import Union
print(os.getcwd())

test_image_base64_1 = encode_image_to_base64("..\\images\\image_1.jpg", resize=1)
test_image_base64_2 = encode_image_to_base64("..\\images\\image_3.png", resize=1)
text_prompt = "The subject is walking in a fantasy landscape with a tree next to him, in the style of Studio Ghibli"




def edit_image_with_qwen_base64(image_base64: Union[str, list], 
                         prompt, 
                         api_key,
                         width=1024,
                         height=1024,
                         seed=4141
                         ):
    """Modifica l'immagine usando Qwen Edit"""
    url = "https://modelslab.com/api/v6/image_editing/qwen_edit"
    
    headers = {"Content-Type": "application/json"}
    print("image_base64:", image_base64[:30]+"...")

    if isinstance(image_base64, list):
        image_1 = image_base64[0]  # take the first image if list
        image_2 = image_base64[1] if len(image_base64) > 1 else None
        image_3 = image_base64[2] if len(image_base64) > 2 else None
        image_4 = image_base64[3] if len(image_base64) > 3 else None
    else:
        image_1 = image_base64
        image_2 = None
        image_3 = None
        image_4 = None

    
    data = {
        "init_image_1": image_1,  # base64 string, not URL
        "init_image_2": image_2,
        "init_image_3": image_3,
        "init_image_4": image_4,
        "prompt": prompt,
        "key": api_key,
        "width": width,
        "height": height,
        "seed": seed,
        "base64": "yes",
        "temp": "yes",
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    update_requests_file(response.json(), urls_file="requests_list.txt")
    return response.json()

def edit_image_with_qwen_edit(images_base64: Union[str, list], 
                         prompt, 
                         api_key,
                         model_id="qwen-edit",
                         width=1024,
                         height=1024,
                         seed=4141
                         ):
    
    """Modifica l'immagine usando Qwen Edit"""
    
    url = "https://modelslab.com/api/v6/image_editing/qwen_edit"
    elegible_models = ["qwen-edit","qwen-edit-2511"]
    
    headers = {"Content-Type": "application/json"}
    

    if isinstance(images_base64, list):
        images = images_base64
    else:
        images = [images_base64]
    
    # star  with data:image or "/9j/"
    images_are_urls = all(img.startswith("https:") for img in images)
    if not images_are_urls:
        print("image_base64:", images[0][:30]+"...")
        use_base64 = "yes"
    else:
        print("image URLs:", images)
        use_base64 = "no"
    
    
    payload = {
        "init_image": images,  # base64 string list or URLs
        "prompt": prompt,
        "model_id": model_id,
        # "num_inference_steps": 8,
        "width": width,
        "height": height,
        "seed": seed,
        "base64": use_base64,
        # "temp": "yes",
    }
    if model_id == "qwen-edit":
        payload.pop("model_id", None)

    if model_id == "qwen-edit-2511_NOOO":
        # add come parametes
        payload["num_inference_steps"] = 8 # Range: 30-50 (2511 funziona meglio con più step)
        payload["guidance_scale"] = 5 # Range: 3.0-5.0 (2511 è sensibile, non andare troppo alto)
        # data["scheduler"] = "DPMSolverMultistepScheduler"
        payload["scheduler"] = "EulerAncestralDiscreteScheduler"
        payload["strength"] = 0.8,  # Range: 0.5-1.0 (quanto modificare l'immagine)
        # data["seed"] = -1 # random seed for more variety

    payload["key"] = api_key
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    update_requests_file(response.json(), urls_file="requests_list.txt")
    return response.json()


def create_image_qwen(local_image_paths: Union[str, list],
                 prompt, 
                 model_id = "qwen-edit-2511",
                 size = None):
    if isinstance(local_image_paths, list):
        image_path = local_image_paths[0]
    else:
        local_image_paths = [local_image_paths]
        image_path = local_image_paths[0]

    try:
        if size in SizeImageDict:
            target_width, target_height = SizeImageDict[size]
            print(f"✓ Target size for aspect ratio {size}: {target_width}x{target_height}")
        else:
            #get with and height
            target_width, target_height, resized_img = resize_image_to_megapixels(image_path, target_mp=1.0)
            print(f"✓ Ridimensionata a: {target_width}x{target_height}")

        # Step 1: Converti in base64
        paths_are_urls = all(path.startswith("https:") for path in local_image_paths)

        if paths_are_urls:
            print("Usando URL immagine direttamente...")
            base64_urls = local_image_paths
        else:
            print("Conversione immagine in base64...")
            base64_images = [encode_image_to_base64(path, resize=1.0) for path in local_image_paths]
            
            base64_urls = [f"data:image/jpeg;base64,{img}" for img in base64_images]

        # Step 3: Modifica con Qwen Edit
        print("Modifica immagine con Qwen...")
        result = edit_image_with_qwen_edit(base64_images, 
                                    prompt, 
                                    api_key,
                                    model_id=model_id,
                                    width=target_width,
                                    height=target_height)

        status = result.get("status", "unknown")
        print(f"✓ Stato richiesta: {status}")
        if status == "processing":
            print(f"Request ID in processing: {result.get('id')}")
            # Step 4: Aggiorna il file dei risultati
            update_requests_file(result, urls_file="requests_list.txt")
            time.sleep(1)
        else:
            print(f" ✗ Modifica immagine fallita. Stato: {status}")
            return None

        return result
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
        print(f"Response: {http_err.response.text}")
        return None
    except Exception as err:
        print(f"Errore: {err}")
        return None


#%%
if __name__ == "__main__":
    if False:
        response = edit_image_with_qwen_base64(
            image_base64=test_image_base64_1,  
            prompt=text_prompt,
            api_key=api_key,
            width=1024,
            height=1024,
            seed=4141
        )

    images = ["..\\images\\image_1.jpg", "..\\images\\image_3.png"]
    # images = ["https://i.pinimg.com/736x/6d/99/8e/6d998ead589a312a4baa00c89c2879e8.jpg"]
    result = create_image_qwen(images, 
                            prompt=text_prompt,
                            model_id="qwen-edit-2511",
                            size = "3:4"
                            )
    
#%%
if __name__ == "__main__":
    result
    data = fetch_image_by_requestid(api_key, result.get("id"))
    print("Fetched id:", data.get("id"))
    data
#%%
if __name__ == "__main__":
    data = fetch_image_by_requestid(api_key, result.get("id"))
    print("Fetched id:", data.get("id"))
    future_url = data.get("future_links", ["no_url"])[0]
    print("Future URL:", future_url)
    if data.get("status") == "success":
            url = data.get("output", ["no_url"])[0]
            show_image_thumbnail(url, size=(200, 200))
    while data.get("status") == "processing":
        data = fetch_image_by_requestid(api_key, result.get("id"))
        print("status:", data.get("status"))
        time.sleep(1)
        
        if data.get("status") == "success":
            show_image_thumbnail(url, size=(200, 200))
            break
#%%

#%%


#%%
"""Crea l'immagine con image reference"""

def request_img2img_v6(images_base64, 
                      prompt, 
                      api_key,
                      negative_prompt=None,
                      model_id = "qwen",
                      width=1024,
                      height=1024,
                      seed=4141,
                      num_inference_steps=8,
                      strength=0.5,
                      temp = "yes",
                      enhance_prompt = "no"
                      ):
    """Crea l'immagine usando Qwen con image reference"""
    elegible_models = [
        # low to medium price models
        "qwen",
               # "num_inference_steps": "8",
               #"strength": "0.5",
        "flux-kontext-dev",
                #"num_inference_steps": "28",
                # "strength": "0.7",
                # "scheduler": "DPMSolverMultistepScheduler",
                # "guidance": "2.5",
                # negative prompt (enhanced): ' (child:1.5), ((((underage)))), ((((child)))), (((kid))), (((preteen))), (teen:1.5) ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy'
                # prompt (enhanced): "hyperrealistic, full body, detailed clothing, highly detailed, cinematic lighting, stunningly beautiful, intricate, sharp focus, f/1. 8, 85mm, (centered image composition), (professionally color graded), ((bright soft diffused light)), volumetric fog, trending on instagram, trending on tumblr, HDR 4K, 8K"
    ]


    url = "https://modelslab.com/api/v6/images/img2img"

    headers = {
        "Content-Type": "application/json"
    }
    if isinstance(images_base64, list):
        images_base64 = images_base64
    else:
        images_base64 = [images_base64]
    if len(images_base64) < 2:
        images_base64 = [ images_base64[0], None]

    print("image_base64:", images_base64[0][:30]+"...")
    
    data = {
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_image": images_base64[0],
            "init_image_2": images_base64[1],
            "width": width,
            "height": height,
            "num_inference_steps": str(num_inference_steps),
            "strength": str(strength),
            "key": api_key,

            "seed": seed,
            "base64": "yes",
            "temp": temp,
            'enhance_prompt': enhance_prompt

        }
    # if model_id == "flux-2-dev":
    #     data.pop("init_image_2", None)
    #     # replace init_image with the image list
    #     data["init_image_1"] = images_base64
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    update_requests_file(response.json(), urls_file="requests_list.txt")
    return response.json()


def create_image_v6(images: Union[str, list],
                 prompt, 
                 model_id = "flux-kontext-dev",
                 strength=0.7,
                 num_inference_steps=28,
                 enhance_prompt="no",
                 negative_prompt=None,
                 size = None):
    if isinstance(images, list):
        local_image_path = images[0]
    else:
        images = [images]
        local_image_path = images[0]    

    try:
        if size in SizeImageDict:
            target_width, target_height = SizeImageDict[size]
            print(f"✓ Target size for aspect ratio {size}: {target_width}x{target_height}")
        else:
            #get with and height
            target_width, target_height, resized_img = resize_image_to_megapixels(local_image_path, target_mp=1.0)
            print(f"✓ Ridimensionata a: {target_width}x{target_height}")

        # Step 1: Converti in base64
        print("Conversione immagine in base64...")
        base64_images = [encode_image_to_base64(path, resize=1.0) for path in images]
        
        base64_urls = [f"data:image/jpeg;base64,{img}" for img in base64_images]

        if model_id == "flux-kontext-dev":
            num_inference_steps = 28
        elif model_id == "qwen":
            num_inference_steps = 8

        # Step 3: Modifica con Model
        print("Modifica immagine con Model...")
        result = request_img2img_v6(base64_images, 
                                    prompt, 
                                    api_key,
                                    model_id=model_id,
                                    width=target_width,
                                    height=target_height,
                                    strength=strength,
                                    num_inference_steps=num_inference_steps,
                                    enhance_prompt=enhance_prompt,
                                    negative_prompt=negative_prompt
                                    )

        status = result.get("status", "unknown")
        print(f"✓ Stato richiesta: {status}")
        if status == "processing":
            print(f"Request ID in processing: {result.get('id')}")
            # Step 4: Aggiorna il file dei risultati
            update_requests_file(result, urls_file="requests_list.txt")
            time.sleep(1)
        else:
            print(f" ✗ Modifica immagine fallita. Stato: {status}")
            return result

        return result
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
        print(f"Response: {http_err.response.text}")
        return None
    except Exception as err:
        print(f"Errore: {err}")
        return None


#%% test
if __name__ == "__main__":
    images = ["..\\images\\image_1.jpg", "..\\images\\image_3.png"]
    images_base64 = [encode_image_to_base64(path, resize=1.0) for path in images]

    if False:
        response = request_img2img_v6(
            images_base64=images_base64,  
            prompt=text_prompt,
            api_key=api_key,
            negative_prompt="((color green)), ((trees and leaves)),((green clothes)), ",
            model_id="flux-kontext-dev",
            width=1024,
            height=1024,
            seed=4141,
            num_inference_steps=28,
            strength=0.7,
            temp="yes",
            enhance_prompt="no"
        )
        print("Response:", response.get("id"), response.get("status"))
    
    result = create_image_v6( images, 
                            prompt=text_prompt,
                            model_id="flux-2-dev",
                            strength=0.7)
#%%
if __name__ == "__main__":
    # result = response

    data = fetch_image_by_requestid(api_key, result.get("id"))
    url = data.get("output", ["no_url"])[0]
    print("Fetched id:", data.get("id"))
    if data.get("status") == "success":
            show_image_thumbnail(url, size=(200, 200))
    while data.get("status") == "processing":
        data = fetch_image_by_requestid(api_key, result.get("id"))
        print("status:", data.get("status"))
        time.sleep(1)
        
        if data.get("status") == "success":
            show_image_thumbnail(url, size=(200, 200))
            break

#%%


def request_img2img_v7(image_url, 
                      prompt, 
                      api_key,
                      model_id = "seedream-4.0-i2i",
                      aspect_ratio="1:1",
                      seed=4141
                      ):
    elegible_models = [
        # low to medium price models

        # high price models
        "seedream-4.0-i2i", 
        "gen4_image_turbo", 
        "flux-2-pro", 
        "nano-banana"
        ]

    url = "https://modelslab.com/api/v7/images/image-to-image"

    headers = {
        "Content-Type": "application/json"
    }
    if isinstance(image_url, list):
        images = image_url
    elif isinstance(image_url, str):
        images = [image_url]
    else:
        images = [
                "https://assets.modelslab.com/uploads/3nveQqFHty5yv8hOYOvgRWxzxLnjnFdG28Mfj5Pd.jpg",
                "https://assets.modelslab.com/uploads/yVgAXzCyWO56bdGQEpBaP1fO6Wk8jYP2O0DoLJeg.jpg",
                "https://assets.modelslab.com/generations/b5aa997f-0784-4d17-913f-8b14df9e6065",
                "https://assets.modelslab.com/generations/b25f10a5-6150-4933-a855-ba3fb51e8fc3"
            ]
    aspect_ratios = ["1:1", "4:3",  "9:16", "16:9", "9:16", "3:2", "2:3", "21:9", "9:21"]
    if aspect_ratio not in aspect_ratios:
        aspect_ratio = "1:1"

    if not prompt:
        prompt = "Hold the Ear rings and chain on her neck, wear the dress to model and put bag in her hand."

    data = {
            "init_image": images,
            "prompt": prompt,
            "model_id": model_id,
            "aspect-ratio": aspect_ratio,
            "key": api_key,
            "seed": seed,
            }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    update_requests_file(response.json(), urls_file="requests_list.txt")
    return response.json()



#%%

if __name__ == "__main__":
    response = create_img2img_v7(
        image_url=[
            "https://assets.modelslab.com/uploads/3nveQqFHty5yv8hOYOvgRWxzxLnjnFdG28Mfj5Pd.jpg",
            "https://assets.modelslab.com/uploads/yVgAXzCyWO56bdGQEpBaP1fO6Wk8jYP2O0DoLJeg.jpg"
        ],
        prompt=None,
        api_key=api_key,
        model_id="seedream-4.0-i2i",
        aspect_ratio="1:1",
        seed=4141
    )
#%%
if __name__ == "__main__":
    print("Response:", response)
    url = response.get("future_links", ["no_url"])[0]
    request_id = response.get("id", "no_id")
    # show_image_thumbnail(url, size=(100, 100))
    save_image_from_base64url(url, folder="img2img_results")
    save_image_from_requestid_base64(url, request_id, folder="img2img_results")
#%%

"""Crea l'immagine"""

#%%
# models pricing
