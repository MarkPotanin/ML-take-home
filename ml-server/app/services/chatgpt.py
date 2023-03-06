from typing import Any, List, Union
from pathlib import Path
from app.models.schemas.classifier import ChatResponseSchema
import openai

class ChatGPT:
    def __init__(self):
        openai.api_key = "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"

    def predict(self, request: str) -> List[ChatResponseSchema]:
        completion = openai.ChatCompletion.create(
                          model="gpt-3.5-turbo",
                          messages=[{"role": "system", "content" : "You’re a kind helpful assistant"},
                            {"role": "user", "content": request}
                          ]
                        )
        result = completion.choices[0].message.content
        return result
