# app/services.py

import os
import uuid
import pandas as pd
from gtts import gTTS
from tensorflow.keras.preprocessing import image as keras_image_processing
import numpy as np
import io
from sentence_transformers import SentenceTransformer
import faiss
import Levenshtein
from googletrans import Translator, LANGUAGES
import traceback
import cv2
# import speech_recognition as sr # Removed

from .core_config import (
    CLASS_NAMES_ENGLISH_IMG_TO_TEXT, ENGLISH_TO_SWAHILI_AUDIO_MAP,
    SWAHILI_TO_ENGLISH_TEXT_MAP, TEXT_TO_VIDEO_CLASS_EXACT_MAP,
    VIDEO_DATA_DIR, VIDEO_DATABASE_CSV, STATIC_FILES_DIR,
    COMBINED_VIDEOS_TEMP_DIR, VIDEO_SEMANTIC_THRESHOLD,
    VIDEO_COMBINE_MAX_CLIP_DURATION_SECONDS, VIDEO_COMBINE_TRANSITION_FRAMES,
    VIDEO_COMBINE_FPS
)

embedding_model_instance = None
video_embeddings_db_instance = None
video_info_list_instance = []
known_words_vocab_instance = set()
google_translator_instance = None

# --- Video Processing Helpers (same as _v10 / _v11_avi_focus) ---
def _read_video_frames_from_path(video_path, max_frames=None, target_w=None, target_h=None):
    if not os.path.exists(video_path):
        print(f"[VIDEO UTIL WARNING] Video file does not exist: {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VIDEO UTIL WARNING] Cannot open video file: {video_path}")
        return []
    frames = []
    count = 0
    while cap.isOpened() and (max_frames is None or count < max_frames):
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_w and target_h and (frame_rgb.shape[1] != target_w or frame_rgb.shape[0] != target_h):
            frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb.astype(np.uint8))
        count += 1
    cap.release()
    return frames

def _add_frames_transition(frames1_clip_end, frames2_clip_start, num_transition_frames):
    if not frames1_clip_end or not frames2_clip_start or num_transition_frames <= 0: return []
    f1_trans_src, f2_trans_src = frames1_clip_end[-num_transition_frames:], frames2_clip_start[:num_transition_frames]
    actual_transitions = min(len(f1_trans_src), len(f2_trans_src), num_transition_frames)
    if actual_transitions == 0: return []
    transition_result_frames = []
    for i in range(actual_transitions):
        alpha = (i + 1) / actual_transitions
        blended = cv2.addWeighted(f1_trans_src[i], 1 - alpha, f2_trans_src[i], alpha, 0)
        transition_result_frames.append(blended.astype(np.uint8))
    return transition_result_frames

def _prepare_combined_video_frames(ordered_video_paths_with_titles):
    all_output_frames = []
    target_w, target_h = None, None
    for path, _ in ordered_video_paths_with_titles:
        if not os.path.exists(path): continue
        cap_dim_check = cv2.VideoCapture(path)
        if cap_dim_check.isOpened():
            target_w = int(cap_dim_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            target_h = int(cap_dim_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_dim_check.release()
            print(f"[VIDEO COMBINE INFO] Target dimensions: {target_w}x{target_h} from '{os.path.basename(path)}'.")
            break
    if not target_w or not target_h: target_w, target_h = 640, 480 
    max_frames_per_clip = int(VIDEO_COMBINE_MAX_CLIP_DURATION_SECONDS * VIDEO_COMBINE_FPS)
    previous_clip_frames_for_transition = []
    for video_idx, (video_path, title) in enumerate(ordered_video_paths_with_titles):
        print(f"[VIDEO COMBINE INFO] Clip {video_idx+1}: '{title}' from '{os.path.basename(video_path)}'")
        current_clip_frames = _read_video_frames_from_path(video_path, max_frames_per_clip, target_w, target_h)
        if not current_clip_frames: continue
        if video_idx > 0 and previous_clip_frames_for_transition:
            transition_frames = _add_frames_transition(previous_clip_frames_for_transition, current_clip_frames, VIDEO_COMBINE_TRANSITION_FRAMES)
            if transition_frames:
                num_to_replace = min(len(transition_frames), len(current_clip_frames))
                all_output_frames.extend(transition_frames[:num_to_replace])
                all_output_frames.extend(current_clip_frames[num_to_replace:])
            else: all_output_frames.extend(current_clip_frames)
        else: all_output_frames.extend(current_clip_frames)
        previous_clip_frames_for_transition = current_clip_frames 
    if not all_output_frames: print("[VIDEO COMBINE ERROR] No frames collected.")
    return all_output_frames, target_w, target_h

def _save_combined_video_to_static(frames_list, width, height):
    if not frames_list:
        print("[VIDEO SAVE DEBUG] No frames provided to save.")
        return None
    base_filename = f"combined_{uuid.uuid4()}"
    options_to_try = [
        {"ext": ".mp4", "fourcc_str": "mp4v", "fourcc_val": cv2.VideoWriter_fourcc(*'m', 'p', '4', 'v')},
        {"ext": ".avi", "fourcc_str": "MJPG", "fourcc_val": cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G')}
    ]
    video_writer, output_path_on_disk, final_filename_for_url, used_option_name = None, None, None, "None"
    for option in options_to_try:
        current_filename = base_filename + option['ext']
        current_output_path = os.path.join(COMBINED_VIDEOS_TEMP_DIR, current_filename)
        os.makedirs(COMBINED_VIDEOS_TEMP_DIR, exist_ok=True)
        writer_instance = cv2.VideoWriter(current_output_path, option["fourcc_val"], VIDEO_COMBINE_FPS, (width, height))
        if writer_instance.isOpened():
            video_writer, output_path_on_disk, final_filename_for_url, used_option_name = writer_instance, current_output_path, current_filename, option['fourcc_str']
            print(f"[VIDEO SAVE INFO] Successfully OPENED VideoWriter with FourCC: {used_option_name} for {option['ext']}")
            break 
        else:
            print(f"[VIDEO SAVE WARNING] FAILED to open VideoWriter with FourCC: {option['fourcc_str']}")
            if os.path.exists(current_output_path) and os.path.getsize(current_output_path) == 0:
                try: os.remove(current_output_path)
                except OSError: pass
    if not video_writer or not video_writer.isOpened():
        print(f"[VIDEO SAVE ERROR] Could not open VideoWriter with any working codecs.")
        return None
    print(f"[VIDEO SAVE INFO] Writing {len(frames_list)} frames to '{output_path_on_disk}' using codec '{used_option_name}'...")
    try:
        for frame_idx, frame_rgb in enumerate(frames_list):
            if frame_rgb is None or frame_rgb.size == 0: continue
            if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[VIDEO SAVE ERROR] Exception while writing frames: {e}"); print(traceback.format_exc())
        video_writer.release()
        if os.path.exists(output_path_on_disk): os.remove(output_path_on_disk)
        return None
    video_writer.release() 
    if os.path.exists(output_path_on_disk) and os.path.getsize(output_path_on_disk) > 0:
        print(f"[VIDEO SAVE INFO] Video saved successfully: '{output_path_on_disk}'")
        return f"/static/combined_temp/{final_filename_for_url}"
    else:
        print(f"[VIDEO SAVE ERROR] Final video file not created or empty: '{output_path_on_disk}'")
        if os.path.exists(output_path_on_disk): os.remove(output_path_on_disk)
        return None

def initialize_text_to_sign_system():
    global embedding_model_instance, video_embeddings_db_instance, video_info_list_instance, known_words_vocab_instance, google_translator_instance
    print("[INFO] Text-to-Sign Service: Initializing system...")
    try:
        df = pd.read_csv(VIDEO_DATABASE_CSV)
        required_cols = ['file_path', 'class']
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] CSV missing required columns: {[c for c in required_cols if c not in df.columns]}.")
            return False
        if 'description' not in df.columns:
            df['description'] = df['class'].astype(str).str.replace('_', ' ').str.replace('/', ' ').str.lower()
        else:
            df['description'] = df['description'].astype(str).str.strip().str.lower()
    except Exception as e:
        print(f"[ERROR] Loading/processing CSV '{VIDEO_DATABASE_CSV}': {e}")
        return False
    print("[INFO] Text-to-Sign: Loading Sentence Transformer...")
    try: embedding_model_instance = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e: print(f"[ERROR] Loading SentenceTransformer: {e}"); embedding_model_instance = None
    video_info_list_instance.clear(); known_words_vocab_instance.clear(); valid_video_count = 0
    descriptions_for_embedding = []
    for index, row in df.iterrows():
        relative_path = str(row['file_path']).strip()
        full_path = os.path.join(VIDEO_DATA_DIR, relative_path)
        if not os.path.exists(full_path):
            print(f"[WARNING] Video in CSV (row {index+2}) not found: '{full_path}'. Skipping.")
            continue
        valid_video_count += 1
        class_name, desc = str(row['class']).strip().replace('/', '_'), str(row['description'])
        video_info_list_instance.append({'description': desc, 'file_path': full_path, 
                                         'relative_path': relative_path, 'class': class_name})
        descriptions_for_embedding.append(desc)
        for token in desc.split() + class_name.lower().replace('_', ' ').split():
            if len(token) > 1: known_words_vocab_instance.add(token)
    print(f"[INFO] Text-to-Sign: Processed {valid_video_count} valid video entries from CSV '{VIDEO_DATABASE_CSV}'.")
    if not descriptions_for_embedding: embedding_model_instance = None 
    if embedding_model_instance and descriptions_for_embedding:
        try:
            embeddings = embedding_model_instance.encode(descriptions_for_embedding, show_progress_bar=False)
            video_embeddings_db_instance = faiss.IndexFlatL2(embeddings.shape[1])
            video_embeddings_db_instance.add(embeddings.astype('float32'))
            print(f"[INFO] FAISS index built: {video_embeddings_db_instance.ntotal} embeddings.")
        except Exception as e: print(f"[ERROR] Building FAISS index: {e}"); video_embeddings_db_instance = None
    else: print("[INFO] Skipping FAISS index (model/descriptions missing).")
    try: google_translator_instance = Translator()
    except Exception as e: print(f"[ERROR] Creating Google Translator: {e}")
    return valid_video_count > 0

def predict_sign_from_image(keras_model, image_bytes):
    try:
        img = keras_image_processing.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
        img_array = keras_image_processing.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        raw_predictions = keras_model.predict(img_array, verbose=0)
        predicted_index = np.argmax(raw_predictions[0])
        predicted_english_text = CLASS_NAMES_ENGLISH_IMG_TO_TEXT[predicted_index]
        confidence = float(raw_predictions[0][predicted_index])
        print(f"[DEBUG] Sign Prediction: Class='{predicted_english_text}', Confidence={confidence:.4f}")
        return predicted_english_text, confidence
    except Exception as e:
        print(f"[ERROR] Sign prediction: {e}"); print(traceback.format_exc())
        return "ErrorInPrediction", 0.0

def translate_english_to_swahili_for_audio(english_text):
    swahili_text = ENGLISH_TO_SWAHILI_AUDIO_MAP.get(english_text)
    if swahili_text: return swahili_text
    if google_translator_instance:
        try:
            if 'sw' not in LANGUAGES: print(f"[WARNING] googletrans no 'sw'.")
            return google_translator_instance.translate(english_text, src='en', dest='sw').text
        except Exception as e: print(f"[ERROR] Google Translate (en->sw) for '{english_text}': {e}")
    return f"{english_text} (Swahili audio map missing or translation failed)"

def generate_swahili_speech(swahili_text):
    try:
        tts = gTTS(text=swahili_text, lang='sw', slow=False)
        filename = f"{uuid.uuid4()}.mp3"
        tts.save(os.path.join(STATIC_FILES_DIR, filename))
        return f"/static/{filename}"
    except Exception as e: print(f"[ERROR] gTTS for '{swahili_text}': {e}"); return None

def translate_text_to_english_for_video_lookup(text, source_language='en'):
    processed = text.lower().strip()
    if source_language == 'en': return processed
    if source_language == 'sw':
        english_eq = SWAHILI_TO_ENGLISH_TEXT_MAP.get(processed)
        if english_eq: 
            print(f"[DEBUG] Swahili to English (map): '{processed}' -> '{english_eq}'")
            return english_eq
        if google_translator_instance:
            try: 
                translated_text = google_translator_instance.translate(processed, src='sw', dest='en').text.lower().strip()
                print(f"[DEBUG] Swahili to English (googletrans): '{processed}' -> '{translated_text}'")
                return translated_text
            except Exception as e: print(f"[ERROR] Google Translate (sw->en) for '{processed}': {e}")
        print(f"[DEBUG] Swahili to English: No map or gtrans fail for '{processed}', returning original.")
        return processed 
    return processed

def _correct_word_with_levenshtein(word):
    if not known_words_vocab_instance or word in known_words_vocab_instance: return word
    min_dist, closest, threshold = float('inf'), word, max(1, int(len(word) * 0.25)) 
    for v_word in known_words_vocab_instance:
        dist = Levenshtein.distance(word, v_word)
        if dist < min_dist: min_dist, closest = dist, v_word
        if dist == 0: break
    return closest if min_dist <= threshold else word

def _semantic_search_for_video_infos(texts, top_n=1):
    if not all([embedding_model_instance, video_embeddings_db_instance, video_info_list_instance]):
        print("[WARNING] Semantic search components not ready."); return []
    try:
        embeddings = embedding_model_instance.encode(texts, show_progress_bar=False)
        distances_batch, indices_batch = video_embeddings_db_instance.search(embeddings.astype('float32'), top_n)
    except Exception as e: print(f"[ERROR] FAISS search: {e}"); return []
    matches = []
    for i in range(len(texts)):
        for j in range(len(indices_batch[i])):
            idx, dist = indices_batch[i][j], distances_batch[i][j]
            if idx != -1 and idx < len(video_info_list_instance) and dist <= VIDEO_SEMANTIC_THRESHOLD:
                matches.append(video_info_list_instance[idx]) 
                if len(matches) >= top_n: break
            elif j == 0: print(f"[INFO] No strong semantic match for '{texts[i]}' (dist: {dist:.2f}).")
        if len(matches) >= top_n: break
    return matches

def find_and_combine_sign_videos(english_text_for_lookup):
    processed_text = english_text_for_lookup.lower().strip()
    ordered_video_infos_to_combine = []
    added_video_abs_paths = set()
    print(f"[DEBUG] Text-to-Sign: Finding videos for processed text: '{processed_text}'")
    dedicated_phrase_video_info = None
    if processed_text != "i love you" and processed_text in TEXT_TO_VIDEO_CLASS_EXACT_MAP:
        target_class_for_phrase = TEXT_TO_VIDEO_CLASS_EXACT_MAP[processed_text]
        print(f"[DEBUG] Checking for dedicated phrase video: Text='{processed_text}' -> Target Class='{target_class_for_phrase}'")
        for vi in video_info_list_instance:
            if vi['class'].lower() == target_class_for_phrase.lower():
                dedicated_phrase_video_info = vi
                print(f"[INFO] Found DEDICATED video for phrase '{processed_text}': Class='{vi['class']}', File='{os.path.basename(vi['file_path'])}'")
                break 
    if dedicated_phrase_video_info:
        ordered_video_infos_to_combine.append(dedicated_phrase_video_info)
        print(f"[INFO] Using single dedicated video for phrase '{processed_text}'.")
    else:
        if processed_text == "i love you":
             print(f"[INFO] For 'i love you', proceeding to word-by-word combination as per preference.")
        else:
            print(f"[INFO] No single dedicated video for phrase '{processed_text}'. Attempting word-by-word combination.")
        words_in_text = processed_text.split()
        for word in words_in_text:
            if len(word) < 2 and word not in ['i', 'a']: continue
            corrected_word = _correct_word_with_levenshtein(word)
            matched_vi_for_word = None
            if corrected_word in TEXT_TO_VIDEO_CLASS_EXACT_MAP:
                target_class = TEXT_TO_VIDEO_CLASS_EXACT_MAP[corrected_word]
                for vi in video_info_list_instance:
                    if vi['class'].lower() == target_class.lower():
                        if vi['file_path'] not in added_video_abs_paths:
                            matched_vi_for_word = vi
                            break 
            if not matched_vi_for_word and embedding_model_instance:
                semantic_matches = _semantic_search_for_video_infos([corrected_word], top_n=1)
                if semantic_matches and semantic_matches[0]['file_path'] not in added_video_abs_paths:
                    matched_vi_for_word = semantic_matches[0]
            if matched_vi_for_word:
                ordered_video_infos_to_combine.append(matched_vi_for_word)
                added_video_abs_paths.add(matched_vi_for_word['file_path'])
                print(f"[DEBUG] Word match: '{word}' (lookup: '{corrected_word}') -> Added video: {os.path.basename(matched_vi_for_word['file_path'])}")
            else:
                print(f"[DEBUG] No video found for word: '{word}' (lookup: '{corrected_word}')")
    if not ordered_video_infos_to_combine:
        print(f"[INFO] No videos found for any components of '{english_text_for_lookup}' after all matching attempts.")
        return None
    video_paths_and_titles_for_combining = [(vi['file_path'], f"{vi['class']}") for vi in ordered_video_infos_to_combine]
    print(f"[INFO] Attempting to combine {len(video_paths_and_titles_for_combining)} video clips for '{english_text_for_lookup}'.")
    combined_frames, w, h = _prepare_combined_video_frames(video_paths_and_titles_for_combining)
    if not combined_frames: 
        print(f"[ERROR] Video frame preparation failed for '{english_text_for_lookup}'.")
        return None
    return _save_combined_video_to_static(combined_frames, w, h)

# Removed transcribe_audio_to_text function as per user request to simplify
