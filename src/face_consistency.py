"""
Massima accuratezza su benchmark / casi difficili: spesso conviene guardare a modelli più recenti o più grandi (ArcFace/varianti, backbone più pesanti, training su dataset più ampi), ma il “meglio” cambia con dataset e dominio (pose/occlusioni/etnie/età).
Detection più robusta: alcuni pipeline moderne preferiscono detector come SCRFD (in InsightFace) al posto di RetinaFace in certi setting (volti piccoli, blur, crowd).
Costo e semplicità: buffalo_l resta spesso una scelta top per applicazioni tipo photo-library (stabile, veloce, buona qualità).
"""
#%%
import json
from huggingface_hub import hf_hub_download
import cv2
import warnings
from insightface.app import FaceAnalysis


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*`estimate` is deprecated since version 0\.26.*",
)


# 1) Leggi token da api_keys.json
with open("api_keys.json", "r", encoding="utf-8") as f:
    HF_TOKEN = json.load(f)["huggingface"]

REPO_ID = "immich-app/buffalo_l"

# 2) Scarica i due ONNX principali (detection + recognition)
det_onnx = hf_hub_download(
    repo_id=REPO_ID,
    filename="detection/model.onnx",
    token=HF_TOKEN,
)
rec_onnx = hf_hub_download(
    repo_id=REPO_ID,
    filename="recognition/model.onnx",
    token=HF_TOKEN,
)

# 3) Usa InsightFace: carica pack buffalo_l (se presente localmente userà i file in cache)
# Nota: FaceAnalysis(name="buffalo_l") è l’API standard per questo pack. [web:7]

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# 4) Estrai embedding da una foto
def get_face_embedding(foto_path: str):
    img = cv2.imread(foto_path)
    faces = app.get(img)

    if not faces:
        print("Nessun volto rilevato")
        return None, faces


    # embedding (tipicamente 512-d); scegli il primo volto o ordina per det_score
    face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
    embedding = face.normed_embedding  # vettore normalizzato (consigliato) [web:13]
    return embedding, faces

foto_path = "outputs/Ian/Sissification/_____feminization_________gemini-2.5-flash-image_20260119_135735.png"
embedding, faces = get_face_embedding(foto_path)

if not faces:
    raise RuntimeError("Nessun volto rilevato")

# embedding (tipicamente 512-d); scegli il primo volto o ordina per det_score
face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
embedding = face.normed_embedding  # vettore normalizzato (consigliato) [web:13]

print(embedding.shape)
print(embedding[:10])

# %%
face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]

# FUNCTIONS COMPARE TWO IMAGES WITH COSINE SIMILARITY GETTING % IF IDENTITY
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_images_cosine_similarity(img_path_1: str, img_path_2: str) -> tuple:
    """
    Compare two images using cosine similarity of face embeddings.
    
    Args:
        img_path_1: Path to first image
        img_path_2: Path to second image
    
    Returns:
        tuple: (similarity_score, similarity_percentage, embedding_1, embedding_2)
            - similarity_score: Raw cosine similarity (-1 to 1)
            - similarity_percentage: Similarity as percentage (0-100%)
            - embedding_1: Face embedding from first image
            - embedding_2: Face embedding from second image
    
    Raises:
        RuntimeError: If no faces detected in either image
    """
    # Get embeddings from both images
    embedding_1, _ = get_face_embedding(img_path_1)
    embedding_2, _ = get_face_embedding(img_path_2)
    
    # Calculate cosine similarity
    # Reshape embeddings to 2D for cosine_similarity function
    if embedding_1 is None or embedding_2 is None:
        return 0, 0, None, None
    
    similarity_score = cosine_similarity(
        embedding_1.reshape(1, -1), 
        embedding_2.reshape(1, -1)
    )[0][0]
    
    # Convert to percentage (normalize from [-1, 1] to [0, 100])
    similarity_percentage = ((similarity_score + 1) / 2) * 100
    
    return similarity_score, similarity_percentage, embedding_1, embedding_2


# Example usage:
score, percentage, emb1, emb2 = compare_images_cosine_similarity(
    "outputs/Ian/Sissification/_____feminization_________gemini-2.5-flash-image_20260119_135735.png",
    "outputs/Ian/refs/20220618_112151.jpg"
)
print(f"Similarity: {percentage:.2f}%")
#%%
# gewt all images paths from outputs/Ian/Sissification/
import glob
image_paths = glob.glob("outputs/Ian/Sissification/*.png")
# image_paths = glob.glob("outputs/Ian/refs/*.*")

results = []
for img_path in image_paths[:10]:
    score, percentage, emb1, emb2 = compare_images_cosine_similarity(
        img_path,
        "outputs/Ian/refs/20220618_112151.jpg"
    )
    print(f"Face Consistency: {percentage:.2f}%")
    results.append((img_path, percentage))
    # print(f"Comparing {img_path} - Similarity: {percentage:.2f}%")
#%%
#SORT RESULTS BY PERCENTAGE DESCENDING
results.sort(key=lambda x: x[1], reverse=True)
for img_path, percentage in results:
    print(f"{img_path}: {percentage:.2f}%")
# %%
