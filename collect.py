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

cap = cv2.VideoCapture(0)
print("starting capture...")

frame_count = 0
c = cap.get(cv2.CAP_PROP_FPS)

def save_frame(frame: npt.NDArray[typing.Any]):
    cur_time = time.time()
    os.makedirs("assets/test", exist_ok=True)
    path = f"./assets/test/{cur_time}.png"
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(path)


while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("frame", frame)

    if frame_count % c == 0:
        save_frame(frame)

    if cv2.waitKey(1) == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

