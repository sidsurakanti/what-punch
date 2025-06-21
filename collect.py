import cv2
import time
import os
from PIL import Image
import typing
from numpy import typing as npt

# get stream
# capture frame every 10 frames
# save img to folder
# clean frames
# classify punches
# get pose landmarks

cap = cv2.VideoCapture("7.mov")
# print("starting capture...")

# frame_count = 0
FPS = cap.get(cv2.CAP_PROP_FPS)
FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_idx = 0

print(FPS, FRAMES)

def save_frame(frame: npt.NDArray[typing.Any]):
    cur_time = time.time()
    os.makedirs("assets/test", exist_ok=True)
    path = f"./assets/test/{cur_time}.png"

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(path)

    return


while cap.isOpened():
    if frame_idx > FRAMES: 
        print("Reached end of video")
        break

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow("f", frame)
    key = cv2.waitKey(0)

    if key == ord('s'):
        save_frame(frame)
        print(f"Saved frame {frame_idx}.")
    elif key == ord('l'):
        frame_idx += 3  
    elif key == ord('h'):
        frame_idx = max(0, frame_idx-1)
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

