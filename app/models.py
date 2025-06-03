# app/models.py

from pydantic import BaseModel, Field
from typing import Optional

class SignToTextResponse(BaseModel):
    predicted_english_text: str
    prediction_confidence: Optional[float] = None # Added for accuracy percentage
    translated_swahili_text: str
    swahili_audio_url: Optional[str] = None 

class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_language: str = Field(default="en")

class TextToSignResponse(BaseModel):
    input_text: str
    processed_text_for_lookup: str
    combined_video_url: Optional[str] = None
    message: Optional[str] = None
