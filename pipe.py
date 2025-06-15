import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision
from pathlib import Path

# download one from here:
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models
model_path = "/Users/sidsurakanti/projects/what-punch/pose_landmarker_full.task"
# model_path = "/"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
running_mode = vision.RunningMode.IMAGE 
load_mp_img = mp.Image.create_from_file 

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=running_mode
)
detector = PoseLandmarker.create_from_options(options)

results = []
folder = Path.cwd() / "assets" / "test"
# folder = Path("assets/test")

for file in folder.glob("*.png"):
    # print(file)
    img = load_mp_img(str(file))
    landmarks = detector.detect(img)
    print(landmarks)
    results.append(landmarks)
    
