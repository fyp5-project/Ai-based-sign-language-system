    # test_opencv_writer.py
import cv2
import numpy as np
import os

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV getBuildInformation: \n{cv2.getBuildInformation()}")


# Define video parameters
filename_mp4 = "test_output.mp4"
filename_avi = "test_output.avi"
width, height = 640, 480
fps = 20
num_frames = 60 # 3 seconds of video

# Create a dummy black frame
frame = np.zeros((height, width, 3), dtype=np.uint8)

# --- Test MP4 with common codecs ---
mp4_codecs_to_test = [
    ('avc1', cv2.VideoWriter_fourcc(*'avc1')), # H.264
    ('X264', cv2.VideoWriter_fourcc(*'X264')), # H.264
    ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))  # MPEG-4
]

print("\n--- Testing MP4 Codecs ---")
for name, fourcc in mp4_codecs_to_test:
    writer = cv2.VideoWriter(filename_mp4, fourcc, fps, (width, height))
    if writer.isOpened():
        print(f"SUCCESS: VideoWriter opened with {name} for MP4.")
        for _ in range(num_frames):
            writer.write(frame)
        writer.release()
        if os.path.exists(filename_mp4) and os.path.getsize(filename_mp4) > 0:
            print(f"  -> Successfully wrote {filename_mp4} using {name}.")
            os.remove(filename_mp4) # Clean up
        else:
            print(f"  -> ERROR: Wrote MP4 with {name}, but file is missing or empty.")
        break # Stop if one MP4 codec works
    else:
        print(f"FAILED: VideoWriter could NOT open with {name} for MP4.")
else: # If no MP4 codec worked
    print("All tested MP4 codecs failed.")


# --- Test AVI with MJPG (often a reliable fallback) ---
print("\n--- Testing AVI with MJPG Codec ---")
fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
writer_avi = cv2.VideoWriter(filename_avi, fourcc_mjpg, fps, (width, height))

if writer_avi.isOpened():
    print("SUCCESS: VideoWriter opened with MJPG for AVI.")
    for _ in range(num_frames):
        writer_avi.write(frame) # Write black frames
    writer_avi.release()
    if os.path.exists(filename_avi) and os.path.getsize(filename_avi) > 0:
        print(f"  -> Successfully wrote {filename_avi} using MJPG.")
        os.remove(filename_avi) # Clean up
    else:
        print(f"  -> ERROR: Wrote AVI with MJPG, but file is missing or empty.")
else:
    print("FAILED: VideoWriter could NOT open with MJPG for AVI.")

print("\n--- Test Complete ---")
    