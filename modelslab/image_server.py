#%%
import http.server
import socketserver
import threading
import os
import requests
from pathlib import Path

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
            return "UNKNOWN_IP"

class ImageServer:
    def __init__(self, port=8000):
        self.port = port
        self.serve_directory = None
        self.server = None
        self.thread = None
        self.is_running = False
        self.external_ip = None
        self.images = []
        self.original_dir = None
        
    def _find_common_directory(self, paths):
        """Trova la directory comune più specifica tra i path"""
        abs_paths = [Path(p).resolve() for p in paths]
        
        # Trova la directory comune
        common = Path(os.path.commonpath(abs_paths))
        
        # Se il common path è un file, prendi la sua directory
        if common.is_file():
            common = common.parent
            
        return common
        
    def start(self, first_image_path):
        """Avvia il server se non è già attivo"""
        if self.is_running:
            return
        
        # Salva directory originale
        self.original_dir = os.getcwd()
        
        # Imposta directory da servire come parent del primo file
        self.serve_directory = Path(first_image_path).resolve().parent
        
        # Cambia directory
        os.chdir(self.serve_directory)
        
        # Trova porta libera
        for attempt in range(10):
            try:
                current_port = self.port + attempt
                socketserver.TCPServer.allow_reuse_address = True
                handler = http.server.SimpleHTTPRequestHandler
                self.server = socketserver.TCPServer(("0.0.0.0", current_port), handler)
                self.port = current_port
                break
            except OSError:
                if attempt == 9:
                    os.chdir(self.original_dir)
                    raise Exception("Nessuna porta disponibile")
                continue
        
        # Avvia server in thread separato
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        # Ottieni IP esterno
        self.external_ip = get_external_ip()
        self.is_running = True
        
        print(f"✓ Server avviato su porta {self.port}")
        print(f"  Directory: {self.serve_directory}")
        print(f"  IP esterno: {self.external_ip}")
        
    def add_image(self, image_path):
        """
        Aggiunge un'immagine al server e restituisce l'URL
        
        Args:
            image_path: str - path locale dell'immagine
        
        Returns:
            str: URL HTTP dell'immagine
        """
        # Converti a Path assoluto
        img_path = Path(image_path).resolve()
        
        if not img_path.exists():
            raise FileNotFoundError(f"File non trovato: {image_path}")
        
        # Avvia server se non attivo (usando la directory del primo file)
        if not self.is_running:
            self.start(img_path)
        
        # Calcola path relativo rispetto alla directory servita
        try:
            rel_path = img_path.relative_to(self.serve_directory)
            filename = str(rel_path).replace('\\', '/')  # Per Windows
        except ValueError:
            # Il file è in una directory diversa - devi riavviare il server
            print(f"⚠ File in directory diversa, riavvio server...")
            self.stop()
            
            # Trova directory comune tra tutti i file già aggiunti e il nuovo
            all_paths = [img['path'] for img in self.images] + [str(img_path)]
            self.serve_directory = self._find_common_directory(all_paths)
            
            # Riavvia server
            self.original_dir = os.getcwd()
            os.chdir(self.serve_directory)
            
            socketserver.TCPServer.allow_reuse_address = True
            handler = http.server.SimpleHTTPRequestHandler
            self.server = socketserver.TCPServer(("0.0.0.0", self.port), handler)
            
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True
            self.thread.start()
            
            self.is_running = True
            print(f"✓ Server riavviato")
            print(f"  Nuova directory: {self.serve_directory}")
            
            # Ricalcola tutti gli URL precedenti
            for img in self.images:
                old_path = Path(img['path'])
                rel = old_path.relative_to(self.serve_directory)
                img['url'] = f"http://{self.external_ip}:{self.port}/{str(rel).replace(chr(92), '/')}"
                print(f"  Aggiornato: {img['filename']} -> {img['url']}")
            
            # Calcola path relativo per il nuovo file
            rel_path = img_path.relative_to(self.serve_directory)
            filename = str(rel_path).replace('\\', '/')
        
        # Crea URL
        url = f"http://{self.external_ip}:{self.port}/{filename}"
        
        self.images.append({
            'filename': img_path.name,
            'url': url,
            'path': str(img_path)
        })
        
        print(f"✓ Immagine aggiunta: {filename}")
        return url
    
    def stop(self):
        """Ferma il server"""
        if not self.is_running:
            print("Server non attivo")
            return
        
        # Ferma server
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        # Torna alla directory originale
        if self.original_dir:
            os.chdir(self.original_dir)
        
        self.is_running = False
        self.server = None
        self.thread = None
        self.images = []
        
        print("✓ Server fermato")
    
    def list_images(self):
        """Mostra tutte le immagini attualmente servite"""
        if not self.images:
            print("Nessuna immagine nel server")
            return []
        
        print(f"\nImmagini attive ({len(self.images)}):")
        for i, img in enumerate(self.images, 1):
            print(f"  {i}. {img['filename']}")
            print(f"     {img['url']}")
        
        return self.images

# Crea istanza globale
image_server = ImageServer(port=9999)

# Funzioni helper semplici
def add_image_to_server(image_path):
    """Aggiunge un'immagine al server e restituisce l'URL"""
    return image_server.add_image(image_path)

def stop_image_server():
    """Ferma il server"""
    image_server.stop()

def list_server_images():
    """Lista tutte le immagini nel server"""
    return image_server.list_images()



if __name__ == "__main__":
    img_path = r"G:\Altri computer\Horizon\horizon_workspace\ai-gen\ai-art\myimagegen\images\image_(1).jpg"
    img_path2 = r"G:\Altri computer\Horizon\horizon_workspace\ai-gen\ai-art\myimagegen\images\image (2).png"
    # Aggiungi più immagini
    url1 = add_image_to_server(img_path)
    url2 = add_image_to_server(img_path2)
    # url3 = add_image_to_server('/path/to/image3.png')

    print(url1)    
    print(url2)    
    # print(url3)  

    # # Oppure con PIL Image
    # from PIL import Image
    # img = Image.open('image.jpg')
    # url4 = add_image_to_server(img)

    # # Oppure con base64
    # import base64
    # with open('image.png', 'rb') as f:
    #     b64_data = base64.b64encode(f.read()).decode()
    # url5 = add_image_to_server(b64_data)

    # Lista immagini attive
    list_server_images()

    # Quando hai finito, ferma il server
    # stop_image_server()

