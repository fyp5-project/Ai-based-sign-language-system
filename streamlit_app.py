# streamlit_app.py

import streamlit as st
import requests
import os
import io
from PIL import Image, UnidentifiedImageError
from typing import Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="centered", page_title="Sign Language Translator")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { text-align: center; color: #1E8449; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #1E8449;
        background-color: #EAFAF1;
        color: #145A32;
        font-weight: bold;
    }
    .stButton>button:hover {
        border-color: #145A32;
        background-color: #D5F5E3;
    }
    .stAlert[data-testid="stAlert"] > div[role="alert"] {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧏🏾 Bi-Directional Sign Language Translator 🗣️")
st.markdown("---")

tab_sign_to_text, tab_text_to_sign = st.tabs([
    "🖼️ Sign Image → Spoken Swahili",
    "📜 Text → Sign Video"
])

# --- Tab 1: Sign Image to Text/Speech ---
with tab_sign_to_text:
    st.header("Translate a Sign Gesture")
    st.write("Upload an image or take a photo of a sign gesture to translate it into spoken Swahili.")
    
    input_method = st.selectbox(
        "Choose input method:", 
        ["📤 Upload Image", "📸 Take Photo"], 
        key="img_input_method_v18"
    )
    
    image_for_display: Optional[object] = None
    image_bytes_for_api: Optional[bytes] = None
    image_filename_for_api: str = "webcam_photo.jpg"
    image_mimetype_for_api: str = "image/jpeg"

    if "Upload Image" in input_method:
        uploaded_image_file = st.file_uploader(
            "Upload an image", 
            type=["png", "jpg", "jpeg"], 
            key="sign_image_uploader_v18"
        )
        if uploaded_image_file:
            try:
                pil_image = Image.open(uploaded_image_file)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                output_bytes_io = io.BytesIO()
                pil_image.save(output_bytes_io, format="JPEG")
                output_bytes_io.seek(0)
                image_for_display = output_bytes_io
                image_bytes_for_api = output_bytes_io.getvalue()
                image_filename_for_api = "uploaded_image.jpg"
                image_mimetype_for_api = "image/jpeg"
            except UnidentifiedImageError:
                st.error("Invalid image file. Please upload a valid PNG, JPG, or JPEG image.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    else:
        camera_photo_buffer = st.camera_input(
            "Take a photo", 
            key="sign_camera_input_v18"
        )
        if camera_photo_buffer:
            try:
                pil_image = Image.open(camera_photo_buffer)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                output_bytes_io = io.BytesIO()
                pil_image.save(output_bytes_io, format="JPEG")
                output_bytes_io.seek(0)
                image_for_display = output_bytes_io
                image_bytes_for_api = output_bytes_io.getvalue()
            except UnidentifiedImageError:
                st.error("Could not process webcam image. Please try again.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    if image_for_display:
        st.image(image_for_display, caption="Your Sign Image", width=400)
        if st.button("Translate this Sign", key="translate_sign_btn_v18", type="primary"):
            if image_bytes_for_api:
                with st.spinner("Analyzing sign..."):
                    files = {"file": (image_filename_for_api, image_bytes_for_api, image_mimetype_for_api)}
                    try:
                        response = requests.post(f"{BACKEND_URL}/sign-to-text", files=files, timeout=45)
                        if response.status_code == 200:
                            result = response.json()
                            st.subheader("Translation Results")
                            st.write(f"**English:** {result['predicted_english_text']}")
                            st.write(f"**Confidence:** {result.get('prediction_confidence', 0.0)*100:.1f}%")
                            st.write(f"**Swahili:** {result['translated_swahili_text']}")
                            if result.get('swahili_audio_url'):
                                audio_full_url = f"{BACKEND_URL.rstrip('/')}{result['swahili_audio_url']}"
                                st.audio(audio_full_url, format='audio/mp3')
                            else:
                                st.warning("Audio unavailable.")
                        else:
                            st.error("Translation failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("No image data available for translation.")
    else:
        st.info("Please provide an image to begin.")

# --- Tab 2: Text to Sign Video ---
with tab_text_to_sign:
    st.header("Generate Sign Language Video")
    st.write("Enter text in English or Swahili to generate a corresponding sign language video.")
    
    source_language = st.selectbox(
        "Input Language:", 
        ("English", "Swahili"), 
        key="text_input_lang_select_v18"
    )
    typed_text = st.text_area(
        "Enter text:", 
        height=100, 
        key="text_to_sign_input_v18", 
        placeholder="e.g., 'I love you' or 'Nakupenda'"
    )
    
    if st.button("Generate Video", key="find_combined_video_btn_v18", type="primary"):
        if typed_text.strip():
            with st.spinner("Generating video..."):
                payload = {
                    "text": typed_text.strip(), 
                    "source_language": 'sw' if source_language == "Swahili" else 'en'
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/text-to-sign", json=payload, timeout=90)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('combined_video_url'):
                            video_url = f"{BACKEND_URL.rstrip('/')}{result['combined_video_url']}"
                            st.video(video_url, format="video/mp4")
                        else:
                            st.warning("Could not generate video.")
                    else:
                        st.error("Video generation failed.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter text to generate a video.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>AI-Powered Sign Language Translation © 2024</div>", unsafe_allow_html=True)