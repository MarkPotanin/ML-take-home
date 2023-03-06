from pydantic import BaseModel, Field


class ClassifierResponseSchema(BaseModel):
    label: str
    score: float

class SimilarityResponseSchema(BaseModel):
    result: dict

class ChatResponseSchema(BaseModel):
    result: str
