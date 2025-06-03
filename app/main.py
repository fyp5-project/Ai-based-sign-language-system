# app/main.py

from fastapi import FastAPI, File, UploadFile # Form removed
from fastapi.responses import JSONResponse, FileResponse 
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input_fn
import tensorflow as tf
from contextlib import asynccontextmanager
import os
import json 
import traceback
import numpy as np
# import shutil # No longer needed for audio uploads
from pathlib import Path 

from .core_config import (
    KERAS_MODEL_PATH, STATIC_FILES_DIR, VIDEO_DATA_DIR, COMBINED_VIDEOS_TEMP_DIR
)
from .models import SignToTextResponse, TextToSignRequest, TextToSignResponse
from .services import (
    predict_sign_from_image, 
    translate_english_to_swahili_for_audio,
    generate_swahili_speech,
    initialize_text_to_sign_system,
    translate_text_to_english_for_video_lookup,
    find_and_combine_sign_videos
    # transcribe_audio_to_text # Removed
)

ml_models = {"keras_sign_model": None} 
model_load_status = {"loaded": False, "error": None, "model_id_at_load": None}

@tf.keras.utils.register_keras_serializable()
class PreprocessInputLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, inputs): return resnet50_preprocess_input_fn(inputs)
    def get_config(self): return super().get_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_models, model_load_status
    print("[INFO] FastAPI Lifespan: Application startup sequence initiated.")
    model_load_status["loaded"] = False; model_load_status["error"] = None
    model_load_status["model_id_at_load"] = None; ml_models["keras_sign_model"] = None
    print("[INFO] Lifespan: Attempting Keras model reconstruction...")
    config_path = os.path.join(KERAS_MODEL_PATH, "config.json")
    weights_path = os.path.join(KERAS_MODEL_PATH, "model.weights.h5") 
    if not os.path.isdir(KERAS_MODEL_PATH) or not os.path.exists(config_path) or not os.path.exists(weights_path):
        model_load_status["error"] = f"Keras model files missing. Dir: {KERAS_MODEL_PATH}"
        print(f"[ERROR] Lifespan: {model_load_status['error']}")
    else:
        try:
            with open(config_path, 'r') as f: model_json = f.read()
            custom_objects = {'PreprocessInputLayer': PreprocessInputLayer}
            model = model_from_json(model_json, custom_objects=custom_objects)
            model.load_weights(weights_path)
            _ = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
            ml_models["keras_sign_model"] = model
            model_load_status["loaded"] = True
            model_load_status["model_id_at_load"] = id(ml_models["keras_sign_model"])
            print(f"[INFO] Lifespan: Keras Sign-to-Text model RECONSTRUCTED. Model ID: {model_load_status['model_id_at_load']}")
        except Exception as e:
            model_load_status["error"] = f"Keras model FAILED: {e}"; print(f"[ERROR] Lifespan: {model_load_status['error']}"); print(traceback.format_exc())
    if not model_load_status["loaded"]: print(f"[WARNING] Lifespan: Keras Sign-to-Text model NOT LOADED. Error: {model_load_status.get('error', 'Unknown')}")
    print("[INFO] Lifespan: Initializing Text-to-Sign system...")
    if initialize_text_to_sign_system(): print("[INFO] Lifespan: Text-to-Sign system ready.")
    else: print("[WARNING] Lifespan: Text-to-Sign system init FAILED/PARTIAL.")
    print("[INFO] FastAPI Lifespan: Application startup sequence complete.")
    yield 
    print("[INFO] FastAPI Lifespan: Shutdown initiated.")
    ml_models.clear(); model_load_status["loaded"] = False
    print("[INFO] Lifespan: Models cleared.")

app = FastAPI(title="Sign Language Translator API", version="1.3.3", lifespan=lifespan) # Incremented version
os.makedirs(STATIC_FILES_DIR, exist_ok=True)
os.makedirs(COMBINED_VIDEOS_TEMP_DIR, exist_ok=True)
os.makedirs(VIDEO_DATA_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static_general") 
app.mount("/videos", StaticFiles(directory=VIDEO_DATA_DIR), name="static_individual_videos") 

@app.get("/", summary="Root")
async def root(): return {"message": "Sign Language Translator API Active"}

@app.post("/sign-to-text", response_model=SignToTextResponse, summary="Sign Image -> Text, Confidence & Swahili Speech")
async def sign_image_to_text_and_speech(file: UploadFile = File(...)):
    keras_model = ml_models.get("keras_sign_model")
    if keras_model is None:
        return JSONResponse(status_code=503, content={"message": f"Sign model unavailable. Startup: {model_load_status.get('error', 'Unknown')}"})
    try:
        image_bytes = await file.read()
        predicted_english, confidence = predict_sign_from_image(keras_model, image_bytes)
        if predicted_english == "ErrorInPrediction":
             return JSONResponse(status_code=500, content={"message": "Prediction error."})
        translated_swahili = translate_english_to_swahili_for_audio(predicted_english)
        swahili_audio_url = generate_swahili_speech(translated_swahili)
        return SignToTextResponse(
            predicted_english_text=predicted_english,
            prediction_confidence=confidence,
            translated_swahili_text=translated_swahili,
            swahili_audio_url=swahili_audio_url
        )
    except Exception as e:
        print(f"[API ERROR] /sign-to-text: {e}"); print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": "Error in sign-to-text."})

@app.post("/text-to-sign", response_model=TextToSignResponse, summary="Text -> Combined Sign Video")
async def text_input_to_sign_video(request: TextToSignRequest):
    try:
        english_lookup = translate_text_to_english_for_video_lookup(request.text, request.source_language)
        combined_video_url = find_and_combine_sign_videos(english_lookup)
        message = f"Input: '{request.text}' (Lookup: '{english_lookup}')."
        if combined_video_url: message += " Combined video created."
        else: message += " Could not generate combined video."
        return TextToSignResponse(input_text=request.text, processed_text_for_lookup=english_lookup, combined_video_url=combined_video_url, message=message)
    except Exception as e:
        print(f"[API ERROR] /text-to-sign: {e}"); print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": "Error in text-to-sign."})

# Removed /speech-to-text endpoint

@app.get("/download-video/{video_filename:path}", summary="Download Combined Video")
async def download_combined_video(video_filename: str):
    file_path = Path(COMBINED_VIDEOS_TEMP_DIR) / video_filename
    print(f"[API /download-video] Attempting to serve: {file_path}")
    if file_path.is_file():
        return FileResponse(
            path=file_path, 
            filename=video_filename, 
            media_type='application/octet-stream' 
        )
    else:
        print(f"[API ERROR] /download-video: File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "Video file not found."})

