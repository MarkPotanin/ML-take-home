import io
import os
import time
from typing import Union, List

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.models.response import ResponseModel, ResponseSimModel,ResponseChatModel
from app.services.classifier import ImageClassifier
from app.services.similarity import ImageSimilarity
from app.services.chatgpt import ChatGPT
from config import CFG

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = CFG.save_path
EMBEDDING_PATH = CFG.train_embeddings_path
Classifier = ImageClassifier(os.path.join(ROOT_DIR,MODEL_PATH))
Similarity = ImageSimilarity(os.path.join(ROOT_DIR,MODEL_PATH), os.path.join(ROOT_DIR,EMBEDDING_PATH))

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to the image classifier server!!, servertime {time.time()}"
    }


@app.post("/classify", tags=["Image Classification"])
async def classify(file: Union[UploadFile, None] = None):
    if not file:
        return ResponseModel(message="No file sent", success=False)

    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = Classifier.predict(image)

    return ResponseModel(data=result, message="Successful classification")

@app.post("/classify_batch", tags=["Batch Classification"])
async def classify_batch(files: Union[List[UploadFile], None] = None):
    if not files:
        return ResponseModel(message="No files sent", success=False)
    images = []
    for file in files:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        images.append(image)
    result = Classifier.predict(images)

    return ResponseModel(data=result, message="Successful classification")

@app.post("/chatgpt", tags=["Some advices from chatgpt"])
async def chatgpt(request: Union[str, None] = None):
    if not request:
        return ResponseModel(message="No request sent", success=False)
    gpt = ChatGPT()
    result = gpt.predict(request)

    return ResponseModel(data=result, message="Advice")


#By default we provide 5 most similar images
@app.post("/find_similar", tags=["Image Similarity"])
async def find_similar(file: Union[UploadFile, None] = None):
    if not file:
        return ResponseSimModel(message="No file sent", success=False)

    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = Similarity.predict(image, 5)
    
    return ResponseSimModel(data=result, message="Here are 5 similar images to the query")


