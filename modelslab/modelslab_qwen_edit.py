#%%
import requests
import json
import base64
import time


# custom scripts
from image_params import resize_image_to_megapixels

def encode_image_to_base64_prefix(image_path):
    """Converte un'immagine locale in base64"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        # Aggiungi il prefisso data URI
        return f"data:image/jpeg;base64,{encoded}"

def encode_image_to_base64(image_path):
    """Converte un'immagine locale in stringa base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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


print("Working directory:", os.getcwd())


def show_folder_images_thumbnails(folder_path):
    """small 100x100 thumbnails of all images in a folder with filename below"""
    import os
    from IPython.display import display, Image

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # display(Image(filename=os.path.join(folder_path, filename), width=100, height=100))
            print(filename)


# in quesot folder , in tuittti i nomi dei file elimina gli mspazi
folder = "../images/fem/bellezze"
import os
for filename in os.listdir(folder):
    if ' ' in filename or '_' in filename:
        new_filename = filename.replace(' ', '_')
        new_filename = new_filename.replace('_', '')
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
        # print(f"Renamed: {filename} -> {new_filename}")

# show_folder_images_thumbnails("../images/")

#%%
import glob
handle = "image"
image_files = glob.glob(f"../images/*{handle}*")



# Percorso della tua immagine locale
local_image_path = image_files[1]

# show selected image
from IPython.display import display, Image
# display(Image(filename=local_image_path))
len(image_files), local_image_path

#%%

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
    return response.json()


# Configurazione
import json
api_key_path = "../api_keys.json"
with open(api_key_path, "r", encoding="utf-8") as f:
    api_keys = json.load(f)
    api_key = api_keys.get("modelslab")
    print("Using ModelsLab API Key:", api_key[:4]+"..."+api_key[-4:])

prompt = "The person is holding a red apple. Professional photography, high detail, sharp focus, professional lighting, 8k"


results = []
for local_image_path in image_files:
    try:
        #get with and height
        from image_params import resize_image_to_megapixels
        new_width, new_height, resized_img = resize_image_to_megapixels(local_image_path, target_mp=1.0)
        print(f"✓ Ridimensionata a: {new_width}x{new_height}")

        # Step 1: Converti in base64
        print("Conversione immagine in base64...")
        base64_image = encode_image_to_base64(local_image_path)
        base64_url = f"data:image/jpeg;base64,{base64_image}"

        # Step 3: Modifica con Qwen Edit
        
        use_base64 = True

        print("Modifica immagine con Qwen...")
        result = edit_image_with_qwen_base64(base64_image, 
                                    prompt, 
                                    api_key,
                                    width=new_width,
                                    height=new_height)

        time.sleep(1)
        result

        results.append(result)
        print("\n✓ Risultato:")
        # from json print remove "init_image" key to avoid huge output
        if "init_image_1" in result:
            json_print = {k: v for k, v in result.items() if not k.startswith("init_image")}
        else:
            json_print = result
        print(json.dumps(json_print, indent=2))
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
        print(f"Response: {http_err.response.text}")
    except Exception as err:
        print(f"Errore: {err}")

#%%

#update results file
update_urls_file_all(results, urls_file="edited_image_urls.txt")

def save_base64_images(results, folder="edited_images"):
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

def save_base64_from_url(img_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    """Salva un'immagine in base64 da un URL in un file locale"""
    import requests
    if img_url:
        response = requests.get(img_url)
        response.raise_for_status()
        base64_data = response.text
        # Rimuovi il prefisso data URI se presente
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(",")[1]
        
        img_data = base64.b64decode(base64_data)
        with open(save_path, "wb") as img_file:
            img_file.write(img_data)
        print(f"✓ Salvata immagine base64 da URL in: {save_path}")
    else:
        print("URL immagine vuoto, impossibile salvare.")

# save_base64_images(results, folder="edited_images")

# show resultds images
print(f"\n✓ Immagini modificate: {len(results)}")
for res in results:
    if True: #res.get("status") != "success":
        img_url = res.get("future_links")[0]
        # img_base64 = res.get("output_base64")[0]
        print(f"Image URL: {img_url}")
        # print(f"Image Base64 (first 100 chars): {img_base64[:10]}...")
        # show thimb 50X50
        timestamp = "_" + str(int(time.time()))
        timestamp = ""
        if use_base64:
            print("Decoding base64 image from URL...")
            # l url contine in base64 -> get base from url content
            # Image URL: https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/temp/5784eebc-54b1-4f85-95ef-ccc2d382046c-0.base64
            # use request to get the
            response = requests.get(img_url)
            base64_data = response.text
            print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")

            if "<!DOCTYPE html>" not in base64_data:
                save_base64_from_url(img_url, save_path=f"edited_images/edited_image_{results.index(res)}{timestamp}.png")
                from IPython.display import display, Image
                # display(Image(data=base64.b64decode(base64_data), width=50, height=50))
        else:
            from IPython.display import display, Image
            display(Image(url=img_url, width=50, height=50))

# results[0]  
#%%
image_files
results

# show dashborad images in ModelsLab using api

import requests
import json


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

# Esempio d'uso
api_key = api_key
request_id = "168491544"  # ID ricevuto quando hai generato l'immagine

result = fetch_queued_image_community(api_key, request_id)
print(result)
result["output"]
# Response example:
# {
#   "status": "success",
#   "id": 13443927,
#   "output": [
#     "https://pub-8b49af329fae499aa563997f5d4068a4.r2.dev/generations/6ef3f81f-14e1-4835-b07a-e00dbe80b6ff-0.png"
#   ]
# }
response = requests.get(result["output"][0])
base64_data = response.text
print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")
display(Image(data=base64.b64decode(base64_data), width=50, height=50))