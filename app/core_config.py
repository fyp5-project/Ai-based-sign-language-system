# app/core_config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras") 

# --- Video Path Configuration ---
# Directory containing your original .MOV, .mp4 etc. files listed in the CSV
ORIGINAL_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "videos") 

# Directory where H.264 MP4 converted versions will be stored
CONVERTED_H264_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "videos_converted_h264")

# The application will serve videos for text-to-sign from this converted directory
VIDEO_DATA_DIR_FOR_SERVING = CONVERTED_H264_VIDEOS_DIR 

VIDEO_DATABASE_CSV = os.path.join(BASE_DIR, "data", "video_database.csv")

STATIC_FILES_DIR = os.path.join(BASE_DIR, "static")
COMBINED_VIDEOS_TEMP_DIR = os.path.join(STATIC_FILES_DIR, "combined_temp")


CLASS_NAMES_ENGLISH_IMG_TO_TEXT = [
    'Church', 'Enough_Satisfied', 'Friend', 'Love', 'Me',
    'Mosque', 'Seat', 'Temple', 'You', 'pray'
]

TEXT_TO_VIDEO_CLASS_EXACT_MAP = {
    "church": "Church", "enough": "Enough_Satisfied", "satisfied": "Enough_Satisfied",
    "friend": "Friend", "love": "Love", "me": "Me", "i": "Me", 
    "mosque": "Mosque", "seat": "Seat", "temple": "Temple", "you": "You",
    "pray": "pray", "hello": "Hello", "goodbye": "Goodbye", "please": "Please",
    "thank you": "Thank_You", "yes": "Yes", "no": "No", "help": "Help",
    "water": "Water", "eat": "Eat", "i love you": "I_Love_You" 
}

SWAHILI_TO_ENGLISH_TEXT_MAP = {
    "kanisa": "church", "imetosha": "enough", "nimeridhika": "satisfied",
    "rafiki": "friend", "upendo": "love", "mimi": "me", "msikiti": "mosque",
    "kiti": "seat", "hekalu": "temple", "wewe": "you", "sala": "pray", "omba": "pray",
    "habari": "hello", "kwaheri": "goodbye", "tafadhali": "please", "asante": "thank you",
    "ndiyo": "yes", "hapana": "no", "msaada": "help", "maji": "water", "kula": "eat",
    "nakupenda": "i love you",
}

ENGLISH_TO_SWAHILI_AUDIO_MAP = {
    'Church': 'Kanisa', 'Enough_Satisfied': 'Imetosha', 'Friend': 'Rafiki',
    'Love': 'Upendo', 'Me': 'Mimi', 'Mosque': 'Msikiti', 'Seat': 'Kiti',
    'Temple': 'Hekalu', 'You': 'Wewe', 'pray': 'Omba',
    'I_Love_You': 'Nakupenda'
}

# --- Video Conversion & Combining Parameters ---
FFMPEG_VIDEO_CODEC = "libx264"  # H.264 codec
FFMPEG_AUDIO_CODEC = "aac"      # Common audio codec for MP4
FFMPEG_CRF = "23"               # Constant Rate Factor (quality)
FFMPEG_PRESET = "medium"        # Encoding speed vs. compression

VIDEO_COMBINE_MAX_CLIP_DURATION_SECONDS = 2 
VIDEO_COMBINE_TRANSITION_FRAMES = 3 
VIDEO_COMBINE_FPS = 20 
VIDEO_SEMANTIC_THRESHOLD = 0.75

# Ensure directories exist
os.makedirs(STATIC_FILES_DIR, exist_ok=True)
os.makedirs(COMBINED_VIDEOS_TEMP_DIR, exist_ok=True)
os.makedirs(ORIGINAL_VIDEOS_DIR, exist_ok=True) # User populates this
os.makedirs(CONVERTED_H264_VIDEOS_DIR, exist_ok=True) # Script populates this

if not os.path.exists(VIDEO_DATABASE_CSV):
    print(f"[CRITICAL WARNING] Core Config: VIDEO_DATABASE_CSV file NOT FOUND at '{VIDEO_DATABASE_CSV}'.")
if not os.path.isdir(KERAS_MODEL_PATH):
    print(f"[CRITICAL WARNING] Core Config: KERAS_MODEL_PATH directory NOT FOUND at '{KERAS_MODEL_PATH}'.")

print(f"[INFO] Core Config: Original videos expected in: {ORIGINAL_VIDEOS_DIR}")
print(f"[INFO] Core Config: Converted H.264 MP4s will be stored in/served from: {CONVERTED_H264_VIDEOS_DIR}")
