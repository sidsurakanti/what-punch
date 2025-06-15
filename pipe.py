import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from pprint import pprint
from PIL import Image
import pandas as pd
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision
from pathlib import Path
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# print(PoseLandmark.__dict__)
# print(PoseLandmark._member_names_)


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

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

  return annotated_image

results = []
folder = Path.cwd() / "assets" / "test"
# folder = Path("assets/test")

for file in sorted(list(folder.glob("*.png"))):
    # img = Image.open(str(file))
    img = load_mp_img(str(file))
    # pprint(type(img).numpy_view.__doc__)
    # pprint(type(img).__doc__)
    
    landmarks = detector.detect(img)
    marks = landmarks.pose_world_landmarks[0] 
    results.append(marks)

    # pprint(type(marks[1]).__dict__)
    
    # print(img.numpy_view().shape)

    alpha_stripped = img.numpy_view()[..., :3]
    bgr_img = cv2.cvtColor(alpha_stripped, cv2.COLOR_RGB2BGR)
    annotated_image = draw_landmarks_on_image(bgr_img, landmarks)

    # cv2.imshow("preview", annotated_image)
    # key = cv2.waitKey(0)  # waits for key press
    # if key == ord('q'): break
    # cv2.destroyAllWindows()  # closes the window

data = []
for res in results:
    data.append([[lm.x, lm.y, lm.z, lm.visibility] for idx, (name, lm) in enumerate(zip(PoseLandmark._member_names_, res))])

print(np.array(data).shape)
data = np.array(data)
idxs = pd.MultiIndex.from_product([np.arange(data.shape[0]), PoseLandmark._member_names_], names=["example", "landmark"])
df = pd.DataFrame(data.reshape(-1, 4), index=idxs, columns=["x", "y", "z", "visibility"])
print(df)


