#%%
import requests
import json
import base64
import time


# custom scripts
from mlslab_utils import *

print("Working directory:", os.getcwd())

# in quesot folder , in tuittti i nomi dei file elimina gli mspazi
folder = "../images/fem/bellezze"
import os
for filename in os.listdir(folder):
    if ' ' in filename or '_' in filename:
        new_filename = filename.replace(' ', '_')
        new_filename = new_filename.replace('_', '')
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
        # print(f"Renamed: {filename} -> {new_filename}")

show_folder_images_thumbnails(folder, max_images=None, thumb_size=(10, 10))

#%%

folder = "../images/"
handle = "image"
image_files = get_images_paths(folder, handle=handle)

# Percorso della tua immagine locale
local_image_path = image_files[1]
print("Length image files:", len(image_files))
# define image batch max 5
image_files = image_files[:5]

# show selected image
from IPython.display import display, Image
# display(Image(filename=local_image_path))
len(image_files), local_image_path

#%%

# Configurazione

prompt = "The person is holding a red apple. Professional photography, high detail, sharp focus, professional lighting, 8k"


results = []

def process_image(local_image_path, prompt):
    try:
        #get with and height
        new_width, new_height, resized_img = resize_image_to_megapixels(local_image_path, target_mp=1.0)
        print(f"✓ Ridimensionata a: {new_width}x{new_height}")

        # Step 1: Converti in base64
        print("Conversione immagine in base64...")
        base64_image = encode_image_to_base64(local_image_path, resize=1.0)
        base64_url = f"data:image/jpeg;base64,{base64_image}"

        # Step 3: Modifica con Qwen Edit
        print("Modifica immagine con Qwen...")
        result = edit_image_with_qwen_base64(base64_image, 
                                    prompt, 
                                    api_key,
                                    width=new_width,
                                    height=new_height)

        status = result.get("status", "unknown")
        print(f"✓ Stato richiesta: {status}")
        results.append(result)

        # Step 4: Aggiorna il file dei risultati
        update_requests_file(result, urls_file="requests_list.txt")
        time.sleep(1)

        return result
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
        print(f"Response: {http_err.response.text}")
        return None
    except Exception as err:
        print(f"Errore: {err}")
        return None

# Pay as you go plan 5 queued API requests
for local_image_path in image_files:
    
    print(f"\n---\nModifica immagine: {local_image_path}")
    result = process_image(local_image_path, prompt)
    max_retries = 3
    retries = 0
    while not result:
        print("Errore nella modifica dell'immagine, riprovo.")
        # retry after 1 second
        time.sleep(1)
        result = process_image(local_image_path, prompt)
        retries += 1
        if retries >= max_retries:
            print("Numero massimo di tentativi raggiunto, salto alla successiva.")
            break

    if result is not None:
        results.append(result)

#%%
for res in results:
    status = res.get("status", "unknown")
    request_id = res.get("id", "unknown")
    # calcualte ime untils status is success
    time_start = time.time()
    while status == "processing":
        print(f"Waiting for request ID {request_id} to complete...")
        time.sleep(1)  # wait 5 seconds before checking again
        # check status again
        updated_res = fetch_queued_image_community(api_key, request_id)
        status = updated_res.get("status", "unknown")

        if status == "success":
            # show future link
            future_link = updated_res.get("output")[0]
            print(f"✓ Request ID {request_id} completed. Future link: {future_link}")
            # show image base64
            response = requests.get(future_link)
            base64_data = response.text
            print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")
            # show thumb 50X50
            from IPython.display import display, Image
            display(Image(data=base64.b64decode(base64_data), width=20, height=20))

            # upadate timed result
            time_end = time.time()
            elapsed_time = time_end - time_start
            print(f"✓ Time taken for request ID {request_id}: {elapsed_time:.2f} seconds")

    print(status)
# 168547166: 107.72 seconds
# success
# 168547174: 5.55 seconds
# success
# 168547180: 11.35 seconds
# success

# total time 124.62 seconds for 3 images, in minutes 2.08 minutes

##########
# ✓ Time taken for request ID 168547743: 17.26 seconds
# ✓ Time taken for request ID 168547743: 1.77 seconds
# ✓ Time taken for request ID 168547752: 5.34 seconds
# ✓ Time taken for request ID 168547752: 1.75 seconds
# ✓ Time taken for request ID 168547756: 10.48 seconds
# ✓ Time taken for request ID 168547756: 1.64 seconds

#%%
############################################################
#update results file
update_urls_file_all(results, urls_file="edited_image_urls.txt")


# save_base64_images_in_results(results, folder="edited_images")

# show resultds images
print(f"\n✓ Immagini modificate: {len(results)}")
for res in results:
    if True: #res.get("status") != "success":
        future_link = res.get("future_links")[0]
        request_id = res.get("id", "unknown")

        # img_base64 = res.get("output_base64")[0]
        print(f"Image URL: {future_link}")
        # print(f"Image Base64 (first 100 chars): {img_base64[:10]}...")
        # show thimb 50X50
        if True:
            print("Decoding base64 image from URL...")
            # l url contine in base64 -> get base from url content
            # Image URL: https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/temp/5784eebc-54b1-4f85-95ef-ccc2d382046c-0.base64
            # use request to get the
            response = requests.get(future_link)
            base64_data = response.text
            print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")

            if "<!DOCTYPE html>" not in base64_data:
                save_base64_from_base64url(future_link, folder=f"edited_images/")
                from IPython.display import display, Image
                # display(Image(data=base64.b64decode(base64_data), width=50, height=50))
        else:
            from IPython.display import display, Image
            display(Image(url=future_link, width=50, height=50))

# results[0]  
#%%
# show dashborad images in ModelsLab using api
import requests
import json

# Esempio d'uso
api_key = api_key
request_id = "168491544"  # ID ricevuto quando hai generato l'immagine

# get list of file in fetch folder
already_fetched = os.listdir("fetched_images")

print(f"\n---\nFetching image for Request ID: {request_id}")
for request_id in get_requestsid_from_file(file_name="requests_list.txt"):
    if request_id != "miss":
        print(f"\n---\nFetching image for Request ID: {request_id}")
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
        # if not already fetched
        if f"fetched_image_{request_id}.png" in already_fetched:
            print(f"✓ Immagine già scaricata per Request ID: {request_id}, salto il download.")
            continue
        if result.get("status") != "failed" and result.get("status") != "processing":
            response = requests.get(result["output"][0])
            base64_data = response.text
            print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")
            display(Image(data=base64.b64decode(base64_data), width=20, height=20))

            # save image
            save_base64_image_from_reqestid(result["output"][0], request_id, folder="fetched_images")

#%%
already_fetched = os.listdir("fetched_images")
already_fetched
#%%
# update_requests_file(results, file_name="requests_list.txt")
get_requestsid_from_file(file_name="requests_list.txt")
