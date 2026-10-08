"""
imgBB Image Uploader con Auto-Cancellazione
=============================================
Carica un'immagine su imgBB e riceve un URL temporaneo.
L'immagine viene auto-cancellata dopo il tempo impostato.

SETUP:
1. Vai su https://api.imgbb.com/
2. Crea un account gratuito
3. Genera la tua API Key
4. Inseriscila qui sotto o passala come variabile d'ambiente IMGBB_API_KEY
"""

import base64
import json
import requests
import os
import sys


# ============================================================
# CONFIGURAZIONE
# ============================================================
# load piakey from api_keys.json if exists, otherwise use environment variable or default placeholder
try:
    with open("api_keys.json", "r") as f:
        keys = json.load(f)
        api_key = keys.get("imgbb", "INSERISCI_QUI_LA_TUA_API_KEY")
        IMGBB_API_KEY = keys.get("imgbb", os.getenv("IMGBB_API_KEY", api_key))
except FileNotFoundError:
        IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "INSERISCI_QUI_LA_TUA_API_KEY")
EXPIRATION_SECONDS = 60  # Auto-cancellazione dopo 60 secondi (min: 60, max: 15552000)
IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"


def upload_to_imgbb(image_path: str, expiration: int = EXPIRATION_SECONDS) -> dict:
    """
    Carica un'immagine su imgBB e restituisce le info dell'upload.

    Args:
        image_path: Percorso locale dell'immagine (jpg, png, gif, bmp, webp)
        expiration: Secondi prima dell'auto-cancellazione (60 - 15552000)

    Returns:
        dict con url, display_url, delete_url, e altri metadati
    """
    # Leggi e converti in base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepara la richiesta
    payload = {
        "key": IMGBB_API_KEY,
        "image": image_base64,
        "expiration": expiration,
    }

    # Opzionale: nome personalizzato
    nome_file = os.path.splitext(os.path.basename(image_path))[0]
    if nome_file:
        payload["name"] = nome_file

    # Upload
    print(f"Caricamento in corso: {image_path}")
    response = requests.post(IMGBB_UPLOAD_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"Upload fallito: {data}")

    info = data["data"]

    print(f"\n{'='*60}")
    print(f"  UPLOAD COMPLETATO!")
    print(f"{'='*60}")
    print(f"  URL immagine:   {info['url']}")
    print(f"  Display URL:    {info['display_url']}")
    print(f"  Delete URL:     {info['delete_url']}")
    print(f"  Dimensione:     {info['size']} bytes")
    print(f"  Scadenza:       {expiration} secondi")
    print(f"{'='*60}\n")

    return info


def upload_base64_to_imgbb(base64_string: str, name: str = "image", expiration: int = EXPIRATION_SECONDS) -> dict:
    """
    Carica un'immagine già in formato base64.
    Utile se l'immagine viene da un'altra fonte (es. canvas, screenshot).
    """
    payload = {
        "key": IMGBB_API_KEY,
        "image": base64_string,
        "name": name,
        "expiration": expiration,
    }

    response = requests.post(IMGBB_UPLOAD_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"Upload fallito: {data}")

    return data["data"]


def upload_url_to_imgbb(image_url: str, name: str = "image", expiration: int = EXPIRATION_SECONDS) -> dict:
    """
    Carica un'immagine a partire da un URL esistente.
    imgBB la scarica e la re-hosta con scadenza.
    """
    payload = {
        "key": IMGBB_API_KEY,
        "image": image_url,
        "name": name,
        "expiration": expiration,
    }

    response = requests.post(IMGBB_UPLOAD_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"Upload fallito: {data}")

    return data["data"]


# ============================================================
# ESEMPIO DI USO CON MODELSLAB
# ============================================================
def esempio_flusso_modelslab(image_path: str, modelslab_api_key: str):
    """
    Esempio completo del flusso:
    1. Upload immagine su imgBB (scade in 60s)
    2. Usa l'URL nella richiesta a ModelsLab
    3. L'immagine si auto-cancella
    """
    # Step 1: Upload su imgBB
    info = upload_to_imgbb(image_path, expiration=120)  # 2 min per sicurezza
    image_url = info["display_url"]

    # Step 2: Usa l'URL con ModelsLab (esempio img2img)
    modelslab_payload = {
        "key": modelslab_api_key,
        "prompt": "Transform into anime, detailed 2D anime style",
        "negative_prompt": "bad quality, blurry",
        "init_image": image_url,  # <-- URL temporaneo da imgBB
        "width": "512",
        "height": "512",
        "samples": 1,
        "safety_checker": False,
    }

    print(f"URL temporaneo per ModelsLab: {image_url}")
    print(f"L'immagine si auto-cancellerà tra 120 secondi.")

    # Decommentare per fare la richiesta effettiva:
    # response = requests.post(
    #     "https://modelslab.com/api/v6/image/img2img",
    #     json=modelslab_payload
    # )
    # print(response.json())

    return image_url


# ============================================================
# MAIN - Esecuzione da riga di comando
# ============================================================
if __name__ == "__main__":
    if IMGBB_API_KEY == "INSERISCI_QUI_LA_TUA_API_KEY":
        print("ERRORE: Inserisci la tua API Key di imgBB!")
        print("  1. Vai su https://api.imgbb.com/")
        print("  2. Crea un account e genera la key")
        print("  3. Sostituiscila nello script o imposta IMGBB_API_KEY come variabile d'ambiente")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso: python imgbb_upload.py <percorso_immagine>")
        print("Esempio: python imgbb_upload.py foto.jpg")
        sys.exit(1)

    percorso = sys.argv[1]

    if not os.path.exists(percorso):
        print(f"ERRORE: File non trovato: {percorso}")
        sys.exit(1)

    risultato = upload_to_imgbb(percorso)
    print(f"Copia questo URL per ModelsLab:\n{risultato['display_url']}")
