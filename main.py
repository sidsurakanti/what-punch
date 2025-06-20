import numpy as np
from numpy import float32
import random
import time
import torch
from math import ceil, floor
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader, dataloader
import matplotlib.pyplot as plt
from datetime import datetime

print(torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = 10 


class Ds(Dataset):
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

trainD = Ds("data.pkl", train=1, augument=True)
testD = Ds("data.pkl", train=0)
trainDL = DataLoader(trainD, batch_size=BATCH_SIZE, shuffle=True)
testDL = DataLoader(testD, batch_size=BATCH_SIZE, shuffle=True) 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(132, 216),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(216, 216),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(216, 4),
        )

    def forward(self, x):
        return self.model(x)


def train(dataloader, model, loss_fn, optimizer):
  model.train()
  
  for batch_i, (X, y) in enumerate(dataloader):
    X, y = X.to(DEVICE), y.to(DEVICE)
    # get predictions
    preds = model(X)
    # calculate loss
    loss = loss_fn(preds, y)
    # backprop
    loss.backward()
    # gradient descent 
    optimizer.step()    
    optimizer.zero_grad()

    if (batch_i % 50 < 1) or (batch_i == len(dataloader) - 1):
      print(f"Batch {batch_i+1}/{len(dataloader)}, Loss: {loss.item():.4f}", end="\r")

def test(dataloader, model, loss_fn):
  model.eval()
  loss_t = correct = 0
  size, num_batches = len(dataloader.dataset), len(dataloader)
  
  # run through testing data
  with torch.no_grad():
    for batch_i, (X, y) in enumerate(dataloader):
      X, y = X.to(DEVICE), y.to(DEVICE)

      # get model preds
      preds = model(X)
      loss_t += loss_fn(preds, y).item()
      correct += (preds.argmax(dim=1) == y).type(torch.float).sum().item()
    
  # calculate average loss & accuracy
  avg_loss = loss_t / num_batches  
  accuracy = correct / size * 100

  print(f"TEST, Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

  return accuracy, avg_loss

def fit(model, epochs: int):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
  loss_fn = torch.nn.CrossEntropyLoss()

  accuracies, losses = [], []

  start_time = datetime.now()

  print("Starting...")
  for epoch in range(epochs):
    print("Epoch", epoch+1)

    train(trainDL, model, loss_fn, optimizer)
    acc, loss = test(testDL, model, loss_fn)

    accuracies.append(acc)
    losses.append(loss)

  torch.save(model.state_dict(), "model_weights.pth")
  print("\nDone!\nWeights saved to 'model_weights.pkl'")
  print(f"Peak Accuracy: {max(accuracies):.2f}% @ Epoch {accuracies.index(max(accuracies))+1}")
  print(f"Time spent training: {(datetime.now() - start_time).total_seconds():.2f}s")
  return accuracies, losses

EPOCHS = 20  
model = Model().to(DEVICE)

acc, loss = fit(model, EPOCHS)

fig, axs = plt.subplots(ncols=2, figsize=(9, 3), layout="constrained")
fig.suptitle("Performance")

epochs = range(1, EPOCHS+1)

ax1, ax2 = axs[0], axs[1]

ax1.plot(epochs, acc, "tab:purple")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epochs")
ax1.grid(True)

ax2.plot(epochs, loss, "tab:orange")
ax2.set_title("Loss")
ax2.set_xlabel("Epochs")
ax2.grid(True)

plt.savefig(f"performance_{int(time.time())}")
