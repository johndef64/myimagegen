#%%
"""
Models 
openai/gpt-audio
openai/gpt-audio-mini
openai/gpt-4o-audio-preview


quick start:

import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json"
  },
  data=json.dumps({
    "model": "openai/gpt-audio",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this audio?"
          },
          {
            "type": "input_audio",
            "input_audio": {
              "data": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB",
              "format": "wav"
            }
          }
        ]
      }
    ]
  })
)
"""
import os
import json
with open("api_keys.json", "r") as f:
    api_keys = json.load(f)
API_KEY = api_keys.get("openrouter", "")

#%%
import requests
import json
import base64
import wave
from pydub import AudioSegment

# Configurazione
# API_KEY = "<OPENROUTER_API_KEY>"  # Inserisci la tua chiave
# SITE_URL = "<YOUR_SITE_URL>"
# SITE_NAME = "<YOUR_SITE_NAME>"

# Parametri audio fissi per gpt-4o-audio-preview (PCM16)
SAMPLE_RATE = 24000  # 24kHz è lo standard per questo modello in output
CHANNELS = 1         # Mono
SAMPLE_WIDTH = 2     # 16-bit = 2 bytes
VOICE = "alloy"  # Voce predefinita
VOICE = "nova"
AUDIO_MODEL = "openai/gpt-4o-audio-preview"
# AUDIO_MODEL = "openai/gpt-audio-mini"
AUDIO_MODEL = "openai/gpt-audio"

SAVE_WAV = False  # Cambia a False per salvare in MP3

PROMPT = {"system": "Parla velocemente ed energicamente.",
          "user": "Dimmi che l'audio streaming funziona perfettamente!"}
PROMPT = {"system": "Parla con un tono sorpreso e stupefatto.",
          "user": "Testo da convertire in audio streaming: Non ci credo! Funziona alla grande! Fantastico!"}

print("Invio richiesta in streaming (formato PCM16)...")

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # "HTTP-Referer": SITE_URL,
        # "X-Title": SITE_NAME,
    },
    data=json.dumps({
        "model": AUDIO_MODEL,
        "modalities": ["text", "audio"],
        "audio": {
            "voice": VOICE,
            "format": "pcm16"  # OBBLIGATORIO con stream=True
        },
        "messages": [
            {
                "role": "system",
                "content": PROMPT["system"]
            },
            {
                "role": "user",
                "content": PROMPT["user"]
            }
        ],
        "stream": True
    }),
    stream=True
)

# Buffer per i dati audio grezzi e il testo
pcm_data = bytearray()
full_transcript = ""

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                try:
                    json_str = decoded_line.replace("data: ", "")
                    chunk = json.loads(json_str)
                    
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0]['delta']
                        
                        # Accumula audio (base64 -> bytes)
                        if 'audio' in delta and 'data' in delta['audio']:
                            # Decodifica il frammento base64 e aggiungilo al buffer
                            pcm_chunk = base64.b64decode(delta['audio']['data'])
                            pcm_data.extend(pcm_chunk)
                        
                        # Accumula transcript
                        if 'audio' in delta and 'transcript' in delta['audio']:
                            full_transcript += delta['audio']['transcript']
                            
                except json.JSONDecodeError:
                    continue
    
    # SALVATAGGIO FILE .WAV
    # PCM16 è raw data. Per ascoltarlo serve un container WAV.
    if len(pcm_data) > 0:
        if SAVE_WAV:
            output_filename = "output_streaming.wav"
            with wave.open(output_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(pcm_data)
            
        # Creazione AudioSegment dai dati raw
        audio = AudioSegment(
            data=pcm_data,
            sample_width=SAMPLE_WIDTH,
            frame_rate=SAMPLE_RATE,
            channels=CHANNELS
        )
        
        # Esportazione in MP3
        prompt_tag = PROMPT["user"][:10].replace(" ", "_").replace(",", "").replace(".", "")
        output_filename = f"output_audio_{VOICE}_{AUDIO_MODEL.replace('/', '_')}_{prompt_tag}.mp3"
        audio.export(output_filename, format="mp3", bitrate="192k")

        print(f"\n✅ Successo! Audio salvato come: {output_filename}")
        print(f"📝 Transcript: {full_transcript}")
    else:
        print("⚠️ Nessun dato audio ricevuto.")

else:
    print(f"Errore API: {response.status_code}")
    print(response.text)

# %%
