from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.model import get_embedding, load_model
from app.utils import find_top_k, load_library

from PIL import Image
import numpy as np
import io

app = FastAPI()
model = load_model()
library_paths, library_embeddings = load_library()

@app.post("/search/")
async def search(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    query_emb = get_embedding(image, model)
    
    results = find_top_k(query_emb, library_embeddings, library_paths)
    return JSONResponse(content=results)