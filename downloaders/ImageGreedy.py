"""
Clipboard Image Downloader

This script provides functionality to download images from the clipboard in multiple formats:
1. Base64 encoded data (data URIs)
2. Direct image data (copied with "Copy Image")
3. Image URLs (copied with "Copy Image Address")

The script scans the clipboard continuously and saves detected images to a specified location (/downloads).
"""

import pyperclip
import base64
import time
import os
from pathlib import Path
import re
from datetime import datetime
from PIL import ImageGrab, Image
import io
import requests
from urllib.parse import urlparse

class ClipboardImageDownloader:
    def __init__(self, download_dir="downloads"):
        """Initialize the downloader with a download directory."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.last_clipboard = ""
        self.last_image_hash = None
        
    def is_base64(self, text):
        """Check if the text is valid Base64."""
        try:
            # Remove data URI prefix if present
            if text.startswith('data:'):
                text = text.split(',', 1)[1] if ',' in text else text
            
            # Base64 pattern check
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(text.strip()):
                return False
            
            # Try to decode
            base64.b64decode(text.strip(), validate=True)
            return len(text.strip()) > 20  # Minimum length check
        except Exception:
            return False
    
    def detect_file_type(self, data):
        """Detect file type from binary data."""
        # Check magic numbers
        if data.startswith(b'\xff\xd8\xff'):
            return 'jpg'
        elif data.startswith(b'\x89PNG'):
            return 'png'
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return 'gif'
        elif data.startswith(b'%PDF'):
            return 'pdf'
        elif data.startswith(b'RIFF') and b'WEBP' in data[:20]:
            return 'webp'
        else:
            return 'bin'
    
    def save_file(self, data, extension, source_type="clipboard"):
        """Save decoded data to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{source_type}_{timestamp}.{extension}"
        filepath = self.download_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(data)
        
        return filepath
    
    def is_image_url(self, text):
        """Check if the text is a valid image URL."""
        try:
            # Check if it's a URL
            result = urlparse(text.strip())
            if not all([result.scheme, result.netloc]):
                return False
            
            # Check if URL ends with image extension
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
            lower_text = text.lower()
            return any(ext in lower_text for ext in image_extensions)
        except Exception:
            return False
    
    def download_image_from_url(self, url):
        """Download image from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url.strip(), headers=headers, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            return None
    
    def get_image_hash(self, image_data):
        """Get a hash of image data for comparison."""
        return hash(image_data[:1000] + image_data[-1000:] if len(image_data) > 2000 else image_data)
    
    def save_pil_image(self, pil_image, source_type="image"):
        """Save PIL Image to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine format
        img_format = pil_image.format if pil_image.format else 'PNG'
        extension = img_format.lower()
        if extension == 'jpeg':
            extension = 'jpg'
        
        filename = f"{source_type}_{timestamp}.{extension}"
        filepath = self.download_dir / filename
        
        # Save image
        pil_image.save(filepath)
        
        return filepath, extension
    
    def monitor_clipboard(self):
        """Continuously monitor clipboard for images (Base64, direct image, URL)."""
        print(f"Monitoring clipboard... Files will be saved to: {self.download_dir.absolute()}")
        print("Supported formats:")
        print("  - Direct images (Copy Image)")
        print("  - Image URLs (Copy Image Address)")
        print("  - Base64 encoded data")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                try:
                    # 1. Check for direct image in clipboard (Copy Image)
                    try:
                        clipboard_image = ImageGrab.grabclipboard()
                        if isinstance(clipboard_image, Image.Image):
                            # Convert to bytes for hashing
                            img_byte_arr = io.BytesIO()
                            clipboard_image.save(img_byte_arr, format=clipboard_image.format or 'PNG')
                            img_bytes = img_byte_arr.getvalue()
                            img_hash = self.get_image_hash(img_bytes)
                            
                            if img_hash != self.last_image_hash:
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Direct image detected!")
                                saved_path, extension = self.save_pil_image(clipboard_image, "direct_image")
                                size = clipboard_image.size
                                print(f"✓ Saved: {saved_path} ({size[0]}x{size[1]}, {len(img_bytes)} bytes)")
                                self.last_image_hash = img_hash
                                self.last_clipboard = ""  # Reset text clipboard
                                time.sleep(0.5)
                                continue
                    except Exception:
                        pass
                    
                    # 2. Check for text content (URL or Base64)
                    clipboard_content = pyperclip.paste()
                    
                    if clipboard_content and clipboard_content != self.last_clipboard:
                        # Check if it's an image URL
                        if self.is_image_url(clipboard_content):
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Image URL detected!")
                            print(f"URL: {clipboard_content[:80]}...")
                            
                            # Download image from URL
                            image_data = self.download_image_from_url(clipboard_content)
                            if image_data:
                                file_type = self.detect_file_type(image_data)
                                saved_path = self.save_file(image_data, file_type, "url_image")
                                print(f"✓ Saved: {saved_path} ({len(image_data)} bytes)")
                                self.last_clipboard = clipboard_content
                                self.last_image_hash = None
                        
                        # Check if it's Base64 data
                        elif self.is_base64(clipboard_content):
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Base64 data detected!")
                            
                            # Remove data URI prefix if present
                            base64_data = clipboard_content
                            if base64_data.startswith('data:'):
                                base64_data = base64_data.split(',', 1)[1] if ',' in base64_data else base64_data
                            
                            # Decode Base64
                            decoded_data = base64.b64decode(base64_data.strip())
                            
                            # Detect file type
                            file_type = self.detect_file_type(decoded_data)
                            
                            # Save file
                            saved_path = self.save_file(decoded_data, file_type, "base64")
                            print(f"✓ Saved: {saved_path} ({len(decoded_data)} bytes)")
                            
                            self.last_clipboard = clipboard_content
                            self.last_image_hash = None
                        
                except Exception as e:
                    print(f"Error processing clipboard: {e}")
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

def main():
    """Main entry point."""
    downloader = ClipboardImageDownloader()
    downloader.monitor_clipboard()

if __name__ == "__main__":
    main()

