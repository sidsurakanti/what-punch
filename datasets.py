from torch.utils.data import Dataset  
from random import shuffle
from math import ceil 
from torchvision.io import decode_image
import torch
from pathlib import Path
import numpy as np

class PunchData(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
        self.paths = ["idle", "jab", "straight", "hook", "uppercut"] 
        # yes it works and yes i wont make it more readable 
        root_path = Path(root)

        data = [
            (file, self.paths.index(folder.name))
            for folder in root_path.iterdir()
            if folder.is_dir() and folder.name in self.paths
            for file in sorted(folder.glob("*.png"))
        ]
        shuffle(data)

        total = len(data) 
        mid = ceil(total * .65)
        self.data = data[:mid] if train else data[mid:]
        self.transforms, self.target_transforms = transform, target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = decode_image(img_path)
        if self.transforms:
            img = self.transforms(img) 
        if self.target_transforms:
            label = self.target_transforms(label)
        return img, label 

class KeypointsData(Dataset):
    # check collect.py 
    def __init__(self, path, train: bool = True, augument: bool = False):
        # shape (examples, features, (xyzv))
        self.augument = augument
        X, y = torch.load(path, weights_only=False)
        self.y = y.astype(float32)
        X = X.reshape(X.shape[0], -1).astype(float32)
        self.total = self.y.shape[0] 
        mid = ceil(self.total * .8)
        self.X = X[:mid] if train else X[mid:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        item = self.X[index]
        label = self.y[index]

        if self.augument:
            item = self.augment_landmarks(item)

        return item, label 
    
    def augment_landmarks(self, landmarks):
        # landmarks: (99,) flattened → reshape to (33, 4)
        landmarks = landmarks.copy().reshape(33, 4)

        if random.random() < 0.5:
            landmarks[:, 0] *= -1  # flip x axis
            landmarks = self.flip_lr_joints(landmarks)

        if random.random() < 0.3:
            noise = np.random.randn(*landmarks.shape) * 0.01
            landmarks += noise

        # asked chat gippity for this ngl 
        if random.random() < 0.3:
            coords = landmarks[:, :3]       
            vis = landmarks[:, 3:]  
            theta = (torch.randn(1).item() * 0.1)
            rot_matrix = np.array([
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta) ]
            ])
            rotated = coords @ rot_matrix   
            landmarks = np.concatenate([rotated, vis], axis=1)

        return landmarks.reshape(-1).astype(float32)

    def flip_lr_joints(self, landmarks):
        LR_PAIRS = [
            (1, 4),   # EYE_INNER
            (2, 5),   # EYE
            (3, 6),   # EYE_OUTER
            (7, 8),   # EAR
            (9, 10),  # MOUTH
            (11, 12), # SHOULDER
            (13, 14), # ELBOW
            (15, 16), # WRIST
            (17, 18), # PINKY
            (19, 20), # INDEX
            (21, 22), # THUMB
            (23, 24), # HIP
            (25, 26), # KNEE
            (27, 28), # ANKLE
            (29, 30), # HEEL
            (31, 32)  # FOOT_INDEX
        ]

        landmarks = landmarks.copy()
        landmarks[:, 0] *= -1  # flip X axis

        for l, r in LR_PAIRS:
            landmarks[[l, r]] = landmarks[[r, l]]

        return landmarks
