# streamlit_app.py

import streamlit as st
import requests
import os
import traceback 
# from st_audiorec import st_audiorec # Removed
import io 

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="wide", page_title="Sign Language Translator")

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
    "📜 Text → Sign Video" # Simplified tab title
])

# --- Tab 1: Sign Image to Text/Speech ---
with tab_sign_to_text:
    st.header("Translate a Sign Gesture")
    
    input_method_img = st.radio(
        "Image input method:", # Simplified label
        ("Upload Image", "Take Photo with Webcam"),
        key="img_input_method_v10",
        horizontal=True
    )

    image_file_buffer = None
    image_filename = "webcam_photo.jpg"

    if input_method_img == "Upload Image":
        uploaded_image_file = st.file_uploader(
            "Select an image file", # Simplified label
            type=["png", "jpg", "jpeg"],
            key="sign_image_uploader_v10",
            label_visibility="collapsed"
        )
        if uploaded_image_file:
            image_file_buffer = uploaded_image_file
            image_filename = uploaded_image_file.name
            
    elif input_method_img == "Take Photo with Webcam":
        camera_photo_buffer = st.camera_input(
            "Point at a sign and click 'Take Photo'", # Simplified label
            key="sign_camera_input_v10"
        )
        if camera_photo_buffer:
            image_file_buffer = camera_photo_buffer
    
    if image_file_buffer:
        col1_img_disp, col2_img_res_disp = st.columns([1, 2])
        with col1_img_disp:
            st.image(image_file_buffer, caption="Your Sign Image", use_container_width=True)
        
        with col2_img_res_disp:
            if st.button("Translate this Sign", key="translate_sign_btn_v10", type="primary"):
                with st.spinner("Analyzing sign..."):
                    files = {"file": (image_filename, image_file_buffer.getvalue(), image_file_buffer.type if hasattr(image_file_buffer, 'type') else 'image/jpeg')}
                    try:
                        response = requests.post(f"{BACKEND_URL}/sign-to-text", files=files, timeout=45)
                        if response.status_code == 200:
                            result = response.json()
                            st.subheader("Translation:")
                            predicted_english = result['predicted_english_text']
                            confidence = result.get('prediction_confidence', 0.0)
                            
                            st.markdown(f"**Interpretation (English):** `{predicted_english}`")
                            st.markdown(f"**Confidence:** `{confidence*100:.1f}%`")
                            st.markdown(f"**Swahili (Spoken):** `{result['translated_swahili_text']}`")
                            
                            if result.get('swahili_audio_url'):
                                audio_full_url = f"{BACKEND_URL.rstrip('/')}{result['swahili_audio_url']}"
                                st.audio(audio_full_url, format='audio/mp3')
                            else: st.warning("Audio could not be generated.")
                        else:
                            err_msg = response.json().get("message", f"Error {response.status_code}")
                            st.error(f"Translation failed: {err_msg}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection Error. Is the backend server running at {BACKEND_URL}? Details: {e}")
                    except Exception as e: 
                        st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please upload an image or take a photo to begin.")


# --- Tab 2: Text to Sign Video ---
with tab_text_to_sign:
    st.header("Generate Sign Language Video from Text")

    # Simplified to only text input
    col_lang_text, col_text_input = st.columns([1, 3])
    with col_lang_text:
        source_language_display_text = st.selectbox("Input Language:", ("English", "Swahili"), key="text_input_lang_select_v10")
    selected_language_code = 'sw' if source_language_display_text == "Swahili" else 'en'
    
    with col_text_input:
        typed_text = st.text_area(f"Enter {source_language_display_text} text:", height=100, key="text_to_sign_input_v10", placeholder=f"e.g., 'I love you' or 'Nakupenda'")
    
    final_text_to_translate = typed_text # Directly use typed_text

    if st.button("Generate Sign Video", key="find_combined_video_btn_v10", type="primary"):
        if final_text_to_translate and final_text_to_translate.strip():
            with st.spinner(f"Generating video for '{final_text_to_translate}'..."):
                payload = {"text": final_text_to_translate.strip(), "source_language": selected_language_code} 
                try:
                    response = requests.post(f"{BACKEND_URL}/text-to-sign", json=payload, timeout=90) 
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('combined_video_url'):
                            st.success(f"Video generated for: '{result['input_text']}'")
                            video_relative_url = result['combined_video_url']
                            display_video_url = f"{BACKEND_URL.rstrip('/')}{video_relative_url}"
                            download_url = f"{BACKEND_URL.rstrip('/')}/download-video{video_relative_url.replace('/static/combined_temp/', '/')}"
                            
                            video_format = "video/x-msvideo" 
                            if video_relative_url.lower().endswith(".mp4"): video_format = "video/mp4"
                            
                            st.video(display_video_url, format=video_format)
                            video_filename = os.path.basename(video_relative_url)
                            st.markdown(f"<small>If playback issues, <a href='{download_url}' download='{video_filename}'>download video ({video_filename})</a>.</small>", unsafe_allow_html=True)
                        else:
                            st.warning(result.get("message", "Could not generate video."))
                    else:
                        err_msg = response.json().get("message", f"Error {response.status_code}")
                        st.error(f"Video generation failed: {err_msg}")
                except requests.exceptions.Timeout:
                    st.error("Video generation timed out.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error. Is backend at {BACKEND_URL} running? Details: {e}")
                except Exception as e: 
                    st.error(f"An unexpected error occurred in Streamlit: {e}")
        else: 
            st.warning("Please enter some text to translate.")
    else: 
        st.info("Enter text to generate a sign video.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>AI-Powered Sign Language Translation © 2024</div>", unsafe_allow_html=True)

