from fastapi import FastAPI, WebSocket
from typing import Tuple
from PIL import Image
import io
import torch
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToImage,
    ToDtype,
    Resize,
)
from model import Model

app = FastAPI()

accel = None
if torch.backends.mps.is_available():
    accel = "mps"
elif torch.cuda.is_available():
    accel = "cuda"
else:
    accel = "cpu"

DEVICE = torch.device(accel)
model = Model().to(DEVICE)
model.load_state_dict(torch.load("etc/weights/idk2.pth", map_location="cpu"))


def inference(img: Image.Image) -> Tuple[float, str]:
    model.eval()

    classes = ["idle", "jab", "straight", "hook", "uppercut"]
    transforms = Compose(
        [
            ToImage(),
            Resize((402, 226)),
            ToDtype(torch.float32, scale=True),  # [0, 255] -> [0, 1] & uint8 -> float32
            Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    img = transforms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        conf = conf.item()
        pred = pred.item()

    return float(conf), classes[int(pred)]


@app.get("/")
def root():
    return {"status": "OK"}


@app.websocket("/predict")
async def predict(websocket: WebSocket):
    await websocket.accept()
    while 1:
        # raw bytes for frame from client
        data = await websocket.receive_bytes()
        img: Image.Image = Image.open(io.BytesIO(data)).convert("RGB")

        conf, pred = inference(img)
        await websocket.send_json(
            {"recieved": True, "prediction": pred, "confidence": conf}
        )
