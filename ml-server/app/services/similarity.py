from typing import Any, List, Union
from pathlib import Path
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, AutoModel
from torchvision import transforms
import numpy as np
from app.models.schemas.classifier import SimilarityResponseSchema

imageType = Union[Image.Image, Any]

class ImageSimilarity:
    def __init__(self,model_path,embedding_path):
        self.model_path = model_path
        self.embedding_path = embedding_path
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path)
        normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        resize = tuple(feature_extractor.size.values())
        self.transforms = transforms.Compose([transforms.Resize(resize),                          
                                               transforms.ToTensor(), normalize
                                               ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        loaded = np.load(self.embedding_path)
        self.all_candidate_embeddings = torch.from_numpy(loaded['emb'])
        self.candidate_ids = loaded['ids']
        self.candidate_paths = loaded['paths']
    def compute_scores(self, emb_one, emb_two):
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
        return scores.numpy().tolist()

    def predict(self, image: imageType,top_k: int) -> SimilarityResponseSchema:
        image_transformed = self.transforms(image).unsqueeze(0)
        new_batch = {"pixel_values": image_transformed.to(self.device)}
        with torch.no_grad():
            query_embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu()
        sim_scores = self.compute_scores(self.all_candidate_embeddings, query_embeddings)
        similarity_mapping = dict(zip(zip(self.candidate_ids,self.candidate_paths), sim_scores))
 
        similarity_mapping_sorted = dict(
            sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
        )
        id_entries = list(similarity_mapping_sorted.keys())[:top_k]

        paths = list(map(lambda x: x[1], id_entries))
        labels = list(map(lambda x: int(x[0].split("_")[-1]), id_entries))
        result = {'paths':paths, 'labels':labels} 
        return result