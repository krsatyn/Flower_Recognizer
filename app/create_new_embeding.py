# Один раз сгенерируй embeddings.npy
import numpy as np
from PIL import Image
import os
from model import load_model, get_embedding
from utils import LIBRARY_DIR

try:
    model = load_model()

    embeddings = []
    image_paths = [os.path.join(LIBRARY_DIR, f) for f in os.listdir(LIBRARY_DIR) if f.endswith(".jpg")]

    all_image_paths = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    for folder in os.listdir(LIBRARY_DIR):
        folder_path = os.path.join(LIBRARY_DIR, folder)
        if os.path.isdir(folder_path):
            all_image_paths += [os.path.join(folder_path, fname)
                                for fname in os.listdir(folder_path)
                                if fname.endswith(".jpg")]

    image_paths = all_image_paths

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        emb = get_embedding(img, model)
        embeddings.append(emb)

    np.save("app/model/embeddings/embeddings.npy", np.array(embeddings))

    print('✅ EMBEDDING CREATE')
    
except:
    print('❌EMBEDDING CREATE ERRORE')