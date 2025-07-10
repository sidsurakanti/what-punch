from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToImage,
    ToDtype,
    Resize,
    RandomAffine,
    RandomHorizontalFlip,
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import PunchData, KeypointsData
from utils import fit, inference

# print(torch.backends.mps.is_available())
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
        # RandomHorizontalFlip(),
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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(
        # nn.Linear(132, 216),
        # nn.Dropout(0.5),
        # nn.ReLU(),
        #
        # nn.Linear(216, 216),
        # nn.Dropout(0.5),
        # nn.ReLU(),
        #
        # nn.Linear(216, 4),
        # )
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 113x113
            nn.Conv2d(16, 32, 3, padding=1),  # 113x113
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 57x57
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 29x29
            nn.Flatten(),
            nn.Linear(28 * 50 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.model(x)


EPOCHS = 20
model = Model().to(DEVICE)
# acc, loss = fit(model, EPOCHS, DEVICE, trainDL, testDL)
model.load_state_dict(torch.load("etc/weights/idk2.pth", map_location="cpu"))
inference(model, DEVICE)
