# app/core_config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# --- CRITICAL PATH CORRECTION ---
# Changed "models_keras" to "models" to match your directory structure
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras") 

VIDEO_DATA_DIR = os.path.join(BASE_DIR, "data", "videos")
VIDEO_DATABASE_CSV = os.path.join(BASE_DIR, "data", "video_database.csv") # Ensure this file exists!

STATIC_FILES_DIR = os.path.join(BASE_DIR, "static")
COMBINED_VIDEOS_TEMP_DIR = os.path.join(STATIC_FILES_DIR, "combined_temp")

CLASS_NAMES_ENGLISH_IMG_TO_TEXT = [
    'Church', 'Enough_Satisfied', 'Friend', 'Love', 'Me',
    'Mosque', 'Seat', 'Temple', 'You', 'pray'
]

# Ensure these class names EXACTLY match the 'class' column in your VIDEO_DATABASE_CSV
# And that corresponding video files exist.
TEXT_TO_VIDEO_CLASS_EXACT_MAP = {
    "church": "Church", 
    "enough": "Enough_Satisfied", # From "Enough/Satisfied" in your CSV sample
    "satisfied": "Enough_Satisfied",
    "friend": "Friend", 
    "love": "Love", # Maps to the single word "Love" video
    "me": "Me", 
    "i": "Me", 
    "mosque": "Mosque", 
    "seat": "Seat", 
    "temple": "Temple", 
    "you": "You",
    "pray": "pray", 
    "hello": "Hello", 
    "goodbye": "Goodbye", 
    "please": "Please",
    "thank you": "Thank_You", 
    "yes": "Yes", 
    "no": "No", 
    "help": "Help",
    "water": "Water", 
    "eat": "Eat", 
    # CRITICAL: This maps the PHRASE "i love you" to a SPECIFIC class "I_Love_You".
    # Your CSV needs a row like: your_i_love_you_video.mp4,I_Love_You,i love you
    "i love you": "I_Love_You" 
}

SWAHILI_TO_ENGLISH_TEXT_MAP = {
    "kanisa": "church", "imetosha": "enough", "nimeridhika": "satisfied",
    "rafiki": "friend", "upendo": "love", "mimi": "me", "msikiti": "mosque",
    "kiti": "seat", "hekalu": "temple", "wewe": "you", "sala": "pray", "omba": "pray",
    "habari": "hello", "kwaheri": "goodbye", "tafadhali": "please", "asante": "thank you",
    "ndiyo": "yes", "hapana": "no", "msaada": "help", "maji": "water", "kula": "eat",
    "nakupenda": "i love you", # This will translate "nakupenda" to "i love you" for lookup
}

ENGLISH_TO_SWAHILI_AUDIO_MAP = {
    'Church': 'Kanisa', 'Enough_Satisfied': 'Imetosha', 'Friend': 'Rafiki',
    'Love': 'Upendo', 'Me': 'Mimi', 'Mosque': 'Msikiti', 'Seat': 'Kiti',
    'Temple': 'Hekalu', 'You': 'Wewe', 'pray': 'Omba',
    'I_Love_You': 'Nakupenda' # If your image model could predict "I_Love_You" class
}

VIDEO_COMBINE_MAX_CLIP_DURATION_SECONDS = 2 
VIDEO_COMBINE_TRANSITION_FRAMES = 3 
VIDEO_COMBINE_FPS = 20 
VIDEO_SEMANTIC_THRESHOLD = 0.75

os.makedirs(STATIC_FILES_DIR, exist_ok=True)
os.makedirs(COMBINED_VIDEOS_TEMP_DIR, exist_ok=True)
os.makedirs(VIDEO_DATA_DIR, exist_ok=True)

if not os.path.exists(VIDEO_DATABASE_CSV):
    print(f"[CRITICAL WARNING] Core Config: VIDEO_DATABASE_CSV file NOT FOUND at '{VIDEO_DATABASE_CSV}'. Text-to-Sign will be severely impaired.")
else:
    print(f"[INFO] Core Config: VIDEO_DATABASE_CSV found at '{VIDEO_DATABASE_CSV}'.")

if not os.path.isdir(KERAS_MODEL_PATH):
    print(f"[CRITICAL WARNING] Core Config: KERAS_MODEL_PATH directory NOT FOUND at '{KERAS_MODEL_PATH}'. Sign-to-Text will fail.")
else:
    print(f"[INFO] Core Config: KERAS_MODEL_PATH set to '{KERAS_MODEL_PATH}'.")

