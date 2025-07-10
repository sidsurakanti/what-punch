from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToDtype,
    Resize,
    RandomAffine,
)
import torch
from torch.utils.data import DataLoader
from datasets import PunchData, KeypointsData
from utils import fit, inference
from model import Model

accel = None
if torch.backends.mps.is_available():
    accel = "mps"
elif torch.cuda.is_available():
    accel = "cuda"
else:
    accel = "cpu"

DEVICE = torch.device(accel)
BATCH_SIZE = 6

# trainD = KeypointsData("data.pkl", train=1, augument=True)
# testD = KeypointsData("data.pkl", train=0)

transforms = Compose(
    [
        Resize((402, 226)),
        RandomAffine(
            (0, 0),
            scale=(0.9, 1.1),
            # translate=(0.15, 0.15),
        ),
        ToDtype(torch.float32, scale=True),  # [0, 255] -> [0, 1] & uint8 -> float32
        Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

trainD = PunchData("data/assets", train=True, transform=transforms)
testD = PunchData("data/assets", train=False, transform=transforms)
trainDL = DataLoader(trainD, batch_size=BATCH_SIZE, shuffle=True)
testDL = DataLoader(testD, batch_size=BATCH_SIZE, shuffle=True)

print("Train data size:", len(trainD))
print("Test data size:", len(testD))


EPOCHS = 20
model = Model().to(DEVICE)
# acc, loss = fit(model, EPOCHS, DEVICE, trainDL, testDL)
model.load_state_dict(torch.load("etc/weights/idk2.pth", map_location="cpu"))
inference(model, DEVICE)
