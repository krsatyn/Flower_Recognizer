import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

LIBRARY_DIR = os.path.join("app", "image_library")
EMBEDDING_FILE = os.path.join("app", "model", "embeddings", "embeddings.npy")

def load_library():
    
    all_image_paths = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    for folder in os.listdir(LIBRARY_DIR):
        folder_path = os.path.join(LIBRARY_DIR, folder)
        if os.path.isdir(folder_path):
            all_image_paths += [os.path.join(folder_path, fname)
                                for fname in os.listdir(folder_path)
                                if fname.endswith(".jpg")]

    image_paths = all_image_paths
    # Эмбеддинги загружаем заранее
    embeddings = np.load(EMBEDDING_FILE)
    

    return image_paths, embeddings
 
#  Функция поиска похожих изображений
def find_top_k(query_emb, db_embs, db_paths, k=5):
    sim_scores = cosine_similarity([query_emb], db_embs)[0]
    top_indices = np.argsort(sim_scores)[::-1][:k]
    
    answer = {}
    for i in top_indices:
        answer[db_paths[i]] = float(sim_scores[i])
    
    return answer
