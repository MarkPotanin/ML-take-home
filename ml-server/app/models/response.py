from typing import Any, List

from app.models.schemas.classifier import ClassifierResponseSchema, SimilarityResponseSchema, ChatResponseSchema


def ResponseModel(
    *, message: str, success=True, data: List[ClassifierResponseSchema] = None
) -> dict:
    return {"success": success, "message": message, "data": data}

def ResponseSimModel(
    *, message: str, success=True, data: SimilarityResponseSchema = None
) -> dict:
    return {"success": success, "message": message, "data": data}

def ResponseChatModel(
    *, message: str, success=True, data: SimilarityResponseSchema = None
) -> dict:
    return {"success": success, "message": message, "data": data}
