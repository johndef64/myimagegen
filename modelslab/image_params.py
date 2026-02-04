import requests
import base64
from io import BytesIO
from PIL import Image
import math


def resize_image_to_megapixels(image_input, target_mp=1.0):
    """
    Carica un'immagine da diverse fonti e la ridimensiona ai megapixel target.
    
    Args:
        image_input: Può essere:
            - str: URL (http/https) o base64 (data:image/...)
            - PIL.Image: Oggetto Image già caricato
            - str: Path locale al file
        target_mp: Megapixel target (default 1.0 MP = 1 milione di pixel)
    
    Returns:
        tuple: (width, height, resized_image)
            - width: Larghezza finale
            - height: Altezza finale
            - resized_image: PIL.Image ridimensionata
    """
    
    # 1. Carica l'immagine in base al tipo di input
    if isinstance(image_input, Image.Image):
        # È già un oggetto PIL
        img = image_input
    
    elif isinstance(image_input, str):
        if image_input.startswith('http://') or image_input.startswith('https://'):
            # URL
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        
        elif image_input.startswith('data:image'):
            # Base64 con data URI
            header, encoded = image_input.split(',', 1)
            image_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(image_data))
        
        else:
            # Assume sia un path locale
            img = Image.open(image_input)
    
    else:
        raise ValueError("Input non supportato. Usa URL, base64, PIL.Image o path locale.")
    
    # 2. Converti in RGB se necessario
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # 3. Calcola dimensioni attuali e target
    current_width, current_height = img.size
    current_pixels = current_width * current_height
    target_pixels = target_mp * 1_000_000
    
    # 4. Se già sotto il target, restituisci dimensioni originali
    # if current_pixels <= target_pixels:
    #     return current_width, current_height, img
    
    # 5. Calcola nuove dimensioni mantenendo aspect ratio
    aspect_ratio = current_width / current_height
    
    # Risolvi: new_width * new_height = target_pixels
    #          new_width / new_height = aspect_ratio
    # Quindi: new_width = sqrt(target_pixels * aspect_ratio)
    new_width = int(math.sqrt(target_pixels * aspect_ratio))
    new_height = int(math.sqrt(target_pixels / aspect_ratio))


    # "message": "The width is invalid. The width must be divisible by 8"
    # Assicura che siano divisibili per 8
    # arrotonda verso il basso
    new_width -= new_width % 8
    new_height -= new_height % 8
    
    # 6. Ridimensiona
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return new_width, new_height, resized_img





#%%
# ============ ESEMPI D'USO ============

if  __name__ == "__main__":
# Esempio 1: Da URL
    url = "https://example.com/image.jpg"
    width, height, img = resize_image_to_megapixels(url, target_mp=2.0)
    print(f"Dimensioni: {width}x{height} (~{(width*height)/1e6:.2f} MP)")

    # Esempio 2: Da file locale
    width, height, img = resize_image_to_megapixels("path/to/image.jpg", target_mp=1.0)
    img.save("output_1mp.jpg", quality=95)

    # Esempio 3: Da base64
    base64_str = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    width, height, img = resize_image_to_megapixels(base64_str, target_mp=0.5)

    # Esempio 4: Da PIL Image
    from PIL import Image
    pil_img = Image.open("input.jpg")
    width, height, img = resize_image_to_megapixels(pil_img, target_mp=1.5)

# Esempio 5: Con info dettagliate
def process_image_with_info(image_input, target_mp=1.0):
    """Versione con output dettagliato"""
    width, height, img = resize_image_to_megapixels(image_input, target_mp)
    
    actual_mp = (width * height) / 1_000_000
    
    print(f"Dimensioni finali: {width}x{height}")
    print(f"Megapixel: {actual_mp:.3f} MP")
    print(f"Aspect ratio: {width/height:.3f}")
    
    return width, height, img

# Usa con ModelsLab
def prepare_for_modelslab(image_input, target_mp=1.0):
    """Prepara immagine per ModelsLab (ridimensiona e converte in base64)"""
    width, height, img = resize_image_to_megapixels(image_input, target_mp)
    
    # Converti in base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}", width, height

# Esempio integrato
# local_image = "my_image.jpg"
# base64_img, w, h = prepare_for_modelslab(local_image, target_mp=1.0)
# print(f"Pronta per upload: {w}x{h}")


#%%



import http.server
import socketserver
import threading
import os

#mi seve una funzoine semplice per ottenere da un immagine in input (path locale,  base64, PIL image) un url http in python


import base64
import requests
from PIL import Image
from io import BytesIO

def image_to_imgur_url(image_input, client_id):
    """
    Carica immagine su Imgur
    
    Args:
        image_input: str (path locale), str (base64), o PIL.Image
        client_id: str - Client ID di Imgur (gratis su https://api.imgur.com/)
    
    Returns:
        str: URL HTTP dell'immagine
    """
    headers = {'Authorization': f'Client-ID {client_id}'}
    
    # Converti a base64
    if isinstance(image_input, str):
        if not image_input.startswith('data:') and len(image_input) < 500:
            with open(image_input, 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            b64_data = image_input.split(',')[1] if ',' in image_input else image_input
    elif isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    response = requests.post(
        'https://api.imgur.com/3/upload',
        headers=headers,
        data={'image': b64_data}
    )
    response.raise_for_status()
    
    return response.json()['data']['link']

import base64
import requests
from PIL import Image
from io import BytesIO

def image_to_url(image_input, api_key):
    """
    Converte un'immagine in URL HTTP usando ImgBB API
    
    Args:
        image_input: str (path locale), str (base64), o PIL.Image
        api_key: str - API key di ImgBB (gratis su https://api.imgbb.com/)
    
    Returns:
        str: URL HTTP dell'immagine caricata
    """
    # Converti input a base64
    if isinstance(image_input, str):
        if image_input.startswith('data:image') or len(image_input) > 500:
            # È già base64
            b64_data = image_input.split(',')[1] if ',' in image_input else image_input
        else:
            # È un path locale
            with open(image_input, 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        # È una PIL Image
        buffer = BytesIO()
        image_input.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Input deve essere path, base64, o PIL Image")
    
    # Upload a ImgBB
    url = 'https://api.imgbb.com/1/upload'
    payload = {
        'key': api_key,
        'image': b64_data
    }
    
    response = requests.post(url, data=payload)
    response.raise_for_status()
    
    return response.json()['data']['url']

# Esempio d'uso
# api_key = 'YOUR_API_KEY'  # Ottieni gratis su https://api.imgbb.com/
# url = image_to_url('path/to/image.png', api_key)
# print(url)



def start_local_server(image_path, port=8000):
    """
    Avvia server HTTP locale e restituisce URL dell'immagine
    (Solo per testing, non per produzione)
    """
    directory = os.path.dirname(os.path.abspath(image_path))
    filename = os.path.basename(image_path)
    
    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    
    # Avvia server in background
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    
    return f"http://localhost:{port}/{filename}"

import http.server
import socketserver
import threading
import os
import requests

def get_external_ip():
    """
    Ottiene l'indirizzo IP pubblico esterno
    """
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        return response.json()['ip']
    except:
        # Fallback
        try:
            response = requests.get('https://ifconfig.me/ip', timeout=5)
            return response.text.strip()
        except:
            raise Exception("Impossibile ottenere IP esterno")

def start_external_server(image_path, port=8000):
    """
    Avvia server HTTP accessibile esternamente e restituisce URL con IP pubblico
    
    Args:
        image_path: str - path all'immagine locale
        port: int - porta del server (default 8000)
    
    Returns:
        str: URL HTTP accessibile esternamente (http://YOUR_IP:PORT/filename.jpg)
    
    Note:
        - Richiede che la porta sia aperta nel firewall/router (port forwarding)
        - Solo per testing, non per uso in produzione
    """
    directory = os.path.dirname(os.path.abspath(image_path))
    filename = os.path.basename(image_path)
    
    os.chdir(directory)
    
    # Bind su 0.0.0.0 per accettare connessioni esterne
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
    
    # Avvia server in background
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    
    # Ottieni IP esterno
    external_ip = get_external_ip()
    
    return f"http://{external_ip}:{port}/{filename}"

# Esempio d'uso
# url = start_external_server('/path/to/image.png', port=8000)
# print(url)  # Output: http://123.45.67.89:8000/image.png

if __name__ == "__main__":
    img_path = r"G:\Altri computer\Horizon\horizon_workspace\ai-gen\ai-art\myimagegen\images\image (1).jpg"
    start_external_server(img_path, port=8787)

#%%%

import http.server
import socketserver
import threading
import os
import requests

def get_external_ip():
    """Ottiene l'indirizzo IP pubblico esterno"""
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        return response.json()['ip']
    except:
        try:
            response = requests.get('https://ifconfig.me/ip', timeout=5)
            return response.text.strip()
        except:
            raise Exception("Impossibile ottenere IP esterno")

# Variabile globale per tenere traccia del server
_server_instance = None

def stop_server():
    """Ferma il server attivo se presente"""
    global _server_instance
    if _server_instance is not None:
        _server_instance.shutdown()
        _server_instance.server_close()
        _server_instance = None
        print("Server precedente fermato")

def start_external_server(image_path, port=8000):
    """
    Avvia server HTTP accessibile esternamente
    
    Args:
        image_path: str - path all'immagine locale
        port: int - porta del server (default 8000)
    
    Returns:
        str: URL HTTP accessibile esternamente
    """
    global _server_instance
    
    # Ferma server precedente se attivo
    stop_server()
    
    directory = os.path.dirname(os.path.abspath(image_path))
    filename = os.path.basename(image_path)
    
    # Prova porte successive se quella richiesta è occupata
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            current_port = port + attempt
            
            # Cambia directory di lavoro
            os.chdir(directory)
            
            # Crea server con riuso indirizzo
            socketserver.TCPServer.allow_reuse_address = True
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(("0.0.0.0", current_port), handler)
            
            _server_instance = httpd
            
            # Avvia server in background
            thread = threading.Thread(target=httpd.serve_forever)
            thread.daemon = True
            thread.start()
            
            # Ottieni IP esterno
            external_ip = get_external_ip()
            external_url = f"http://{external_ip}:{current_port}/{filename}"
            
            if attempt > 0:
                print(f"Porta {port} occupata, usando porta {current_port}")
            
            print(f"Server avviato su porta {current_port}")
            return external_url
            
        except OSError as e:
            if e.winerror == 10048 and attempt < max_attempts - 1:
                # Porta occupata, prova la successiva
                continue
            else:
                raise Exception(f"Impossibile avviare server dopo {max_attempts} tentativi")
    
    raise Exception("Tutte le porte sono occupate")

if __name__ == "__main__":
    # Esempio d'uso
    url = start_external_server(img_path, port=8000)
    print(url)

    # Per fermare manualmente il server
    stop_server()

#%%