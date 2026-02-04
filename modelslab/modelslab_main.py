#%%
import requests
import json
import base64
import time


# custom scripts
from mlslab_utils import *

print("Working directory:", os.getcwd())

# in quesot folder , in tuittti i nomi dei file elimina gli mspazi

folder = "../images/"
handle = "image"
image_files = get_images_paths(folder, handle=handle)

folder = "../images/fem/bellezze"
# folder = "../images\Posers\Full Body (ref)"

import os
def clean_filenames_in_folder(folder):
    for filename in os.listdir(folder):
        if ' ' in filename or '_' in filename:
            new_filename = filename.replace(' ', '_')
            # new_filename = new_filename.replace('_', '')
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
            # print(f"Renamed: {filename} -> {new_filename}")

show_folder_images_thumbnails(folder, max_images=5, thumb_size=(10, 10))

#%%

# handle = "814659020137855813"
image_files = get_images_paths(folder, handle=handle)


image_files = get_images_paths(folder, handle=f"{handle} - DQy3sbHDLgT")
image_files = get_images_paths(folder, handle=handle)

# Percorso della tua immagine locale
local_image_path = image_files[0]
print("Length image files:", len(image_files))
# define image batch max 5
image_files = image_files[:5]

ref_image_path = get_images_paths("../images/sketches", handle="")

# show selected image
from IPython.display import display, Image
# display(Image(filename=local_image_path))
len(image_files), local_image_path
#%%

# Configurazione
prompt = "The person is holding a red apple. Professional photography, high detail, sharp focus, professional lighting, 8k"



SizeImageDict = {
# square
"1:1": (1024, 1024),
# vertical
"3:4": (768, 1024),
"1:2": (512, 1024),
"9:16": (576, 1024),
#landscape
"4:3": (1024, 768),
"2:1": (1024, 512),
"16:9": (1024, 576),   
}

def create_image_qwen_retry(*args, **kwargs):
    max_retries = 5
    retries = 0
    result = None
    while not result:
        result = create_image_qwen(*args, **kwargs)
    
        retries += 1
        time.sleep(1)  # wait before retrying
        if retries >= max_retries:
            print("Numero massimo di tentativi raggiunto, salto alla successiva.")
            break
    return result

def create_image_v6_retry(*args, **kwargs):
    max_retries = 5
    result = None
    while not result:
        result = create_image_v6(*args, **kwargs)
    
        retries += 1
        time.sleep(1)  # wait before retrying
        if retries >= max_retries:
            print("Numero massimo di tentativi raggiunto, salto alla successiva.")
            break
    return result



# Pay as you go plan 5 queued API requests
results = []
for local_image_path in image_files[:1]:
    MODEL     = "qwen-edit"
    MODEL     = "flux-kontext-dev"
    MODEL     = "qwen-edit-2511"
    # MODEL     = "flux-2-dev"
    
    print(f"\n---\nModifica immagine: {local_image_path}")
    if MODEL in ["flux-kontext-dev", "qwen", "flux-2-dev"]:
        result = create_image_v6_retry(local_image_path,
                                    prompt, 
                                    model_id=MODEL
                                    )
        if result is not None:
            results.append(result)

    else:
        print("Using Qwen model for image editing")
        for MODEL in ["qwen-edit-2511", "qwen-edit"]:
            result = create_image_qwen_retry(local_image_path, prompt, model_id=MODEL)
            if result is not None:
                results.append(result)
            time.sleep(1)  # wait before next request
        

fetch_image_by_requestid(api_key, result.get("id"))
#%%
for res in results:
    data = fetch_image_by_requestid(api_key, res.get("id"))
    print(data.get("id"), data.get("status"), data.get("future_links"))
#%%
for res in results:
    data = fetch_image_by_requestid(api_key, res.get("id"))
    status = data.get("status", "unknown")
    request_id = data.get("id", "unknown")
    print(f"Request ID: {request_id}, Status: {status}")
    if status == "success":
        output_link = data.get("output")[0]
        # show image base64
        response = requests.get(output_link)
        base64_data = response.text
        from IPython.display import display, Image
        display(Image(data=base64.b64decode(base64_data), width=20, height=20))
#%%

#%%  %%%%%%%%%%%%
print("Results length:", len(results))
for res in results:
    status = res.get("status", "unknown")
    request_id = res.get("id", "unknown")
    # calcualte ime untils status is success
    time_start = time.time()
    while status == "processing":
        print(f"Waiting for request ID {request_id} to complete...")
        time.sleep(1)  # wait 5 seconds before checking again
        # check status again
        updated_res = fetch_image_by_requestid(api_key, request_id)
        status = updated_res.get("status", "unknown")

        if status == "success":
            # show future link
            output_link = updated_res.get("output")[0]
            print(f"✓ Request ID {request_id} completed. Future link: {output_link}")
            # show image base64
            response = requests.get(output_link)
            base64_data = response.text
            print(f"Image Base64 (first 100 chars): {base64_data[:100]}...")
            # show thumb 50X50
            from IPython.display import display, Image
            display(Image(data=base64.b64decode(base64_data), width=20, height=20))
            # save image
            save_image_from_requestid_base64(output_link, request_id, folder="../images/sketches")

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
                save_image_from_base64url(future_link, folder=f"edited_images/")
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
            save_image_from_requestid_base64(result["output"][0], request_id, folder="fetched_images")

#%%
already_fetched = os.listdir("fetched_images")
already_fetched
#%%
# update_requests_file(results, file_name="requests_list.txt")
get_requestsid_from_file(file_name="requests_list.txt")
