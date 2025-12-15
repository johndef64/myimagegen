"""
Docstring for downloadres.clipboard_base63_downloader

This script provides functionality to download resources encoded in Base64 format from the clipboard.

This script scans the clipboard for Base64 encoded data, decodes it, and saves it as an image to a specified location (/downloads) automatically every time the clipboard contains a new Base64 encoded resource.
"""

import pyperclip
import base64
import time
import os
from pathlib import Path
import re
from datetime import datetime

class ClipboardBase64Downloader:
    def __init__(self, download_dir="downloads"):
        """Initialize the downloader with a download directory."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.last_clipboard = ""
        
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
    
    def save_file(self, data, extension):
        """Save decoded data to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"clipboard_{timestamp}.{extension}"
        filepath = self.download_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(data)
        
        return filepath
    
    def monitor_clipboard(self):
        """Continuously monitor clipboard for Base64 data."""
        print(f"Monitoring clipboard... Files will be saved to: {self.download_dir.absolute()}")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                try:
                    clipboard_content = pyperclip.paste()
                    
                    # Check if clipboard content changed and is Base64
                    if clipboard_content != self.last_clipboard and self.is_base64(clipboard_content):
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New Base64 data detected!")
                        
                        # Remove data URI prefix if present
                        base64_data = clipboard_content
                        if base64_data.startswith('data:'):
                            base64_data = base64_data.split(',', 1)[1] if ',' in base64_data else base64_data
                        
                        # Decode Base64
                        decoded_data = base64.b64decode(base64_data.strip())
                        
                        # Detect file type
                        file_type = self.detect_file_type(decoded_data)
                        
                        # Save file
                        saved_path = self.save_file(decoded_data, file_type)
                        print(f"✓ Saved: {saved_path} ({len(decoded_data)} bytes)")
                        
                        self.last_clipboard = clipboard_content
                        
                except Exception as e:
                    print(f"Error processing clipboard: {e}")
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

def main():
    """Main entry point."""
    downloader = ClipboardBase64Downloader()
    downloader.monitor_clipboard()

if __name__ == "__main__":
    main()

