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
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = int(time.time())
    
    for i, res in enumerate(results):
        if True: #res.get("status") == "success":
            for j, img_base64 in enumerate(res.get("output_base64", [])):
                # Rimuovi il prefisso data URI se presente
                if img_base64.startswith("data:image"):
                    img_base64 = img_base64.split(",")[1]
                
                img_data = base64.b64decode(img_base64)
                file_path = os.path.join(folder, f"edited_image_{i}_{j}_{timestamp}.png")
                with open(file_path, "wb") as img_file:
                    img_file.write(img_data)
                print(f"✓ Salvata immagine base64 in: {file_path}")

def save_base64_from_base64url(img_url, folder):
    os.makedirs(os.path.dirname(folder), exist_ok=True)
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

        file_path = os.path.join(folder, f"edited_image_{url_id}.png")

        with open(file_path, "wb") as img_file:
            img_file.write(img_data)
        print(f"✓ Salvata immagine base64 da URL in: {file_path}")
    else:
        print("URL immagine vuoto, impossibile salvare.")

def fetch_queued_image_community(api_key, request_id):
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

def save_base64_image_from_reqestid(img_url, request_id, folder="fetched_images"):
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

def show_folder_images_thumbnails(folder_path):
    """small 100x100 thumbnails of all images in a folder with filename below"""
    import os
    from IPython.display import display, Image

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # display(Image(filename=os.path.join(folder_path, filename), width=100, height=100))
            print(filename)


# file manaegr
   
def get_images_paths(folder, handle=""):
    import glob
    image_files = glob.glob(f"{folder}/*{handle}*")
    return image_files



############ MOdels FUNCTIONS ############


"""Modifica l'immagine usando Qwen Edit"""

# Rest API example request payload
# json_request = {
#   "prompt": "A girl is showing her perfect squared red nails to the camera",
#   "webhook": null,
#   "width": 1024,
#   "height": 1024,
#   "samples": 1,
#   "num_inference_steps": 8,
#   "seed": 4141,
#   v
#   "safety_checker": "no",
#   "safety_checker_type": "black",
#   "track_id": null,
#   "base64": "no",
#   "watermark": "no",
#   "init_image_1": "https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/4500897471769798941.png",
#   "init_image_2": null,
#   "init_image_3": null,
#   "init_image_4": null,
#   "id": 168279749
# }


def edit_image_with_qwen_base64(image_base64, 
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
    
    data = {
        "init_image": image_base64,  # base64 string, not URL
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


"""Crea l'immagine usando Qwen con image reference"""



"""Crea l'immagine usando Qwen"""