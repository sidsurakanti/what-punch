import torch
import matplotlib.pyplot as plt
from collections import deque, Counter
from pathlib import Path
import time
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype, Resize, RandomAffine, RandomHorizontalFlip
from datetime import datetime
import cv2

def inference(model, device):
    model.eval()
    # get capture
    classes = ["idle", "jab", "straight", "hook", "uppercut"]
    cap = cv2.VideoCapture("4.mov")
    # resize and apply main transforms
    transforms = Compose([
        ToImage(),
        Resize((402, 226)),
        ToDtype(torch.float32, scale=True), # [0, 255] -> [0, 1] & uint8 -> float32
        Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    frame_idx = 0
    pred_buffer = deque(maxlen=5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("f", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if frame_idx % 4 > 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred_buffer.append(classes[pred])
        
            # print(conf, pred)
            if conf > .5:
                print(Counter(pred_buffer).most_common(1)[0][0])


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for batch_i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # get predictions
        preds = model(X)
        # calculate loss
        loss = loss_fn(preds, y)

        # backprop
        loss.backward()
        # gradient descent 
        optimizer.step()    
        optimizer.zero_grad()

        if (batch_i % 4 < 1) or (batch_i == len(dataloader) - 1):
            print(f"BATCH {batch_i+1}/{len(dataloader)}, LOSS: {loss.item():.4f}", end="\r")


def test(dataloader, model, loss_fn, device):
    model.eval()
    loss_t = correct = 0
    size, num_batches = len(dataloader.dataset), len(dataloader)

    # run through testing data
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # get model preds
            preds = model(X)
            loss_t += loss_fn(preds, y).item()
            correct += (preds.argmax(dim=1) == y).type(torch.float).sum().item()

    # calculate average loss & accuracy
    avg_loss = loss_t / num_batches  
    accuracy = correct / size * 100

    print(f"TEST, ACCURACY: {accuracy:.2f}%, LOSS: {avg_loss:.4f}")

    return accuracy, avg_loss


def fit(model, epochs: int, device, trainDL, testDL):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    accuracies, losses = [], []
    start_time = datetime.now()

    print("STARTING...")
    for epoch in range(epochs):
        print("EPOCH", epoch+1, ">>>", end="\n")

        train(trainDL, model, loss_fn, optimizer, device)
        acc, loss = test(testDL, model, loss_fn, device)

        accuracies.append(acc)
        losses.append(loss)
    
    if input("Save model weights? (y/n) >>> ").lower() in ("y", "yes"):
        name = input("Save model weights to? >>> ")
        torch.save(model.state_dict(), f"{ name }.pth")
        print(f"\nDone! Weights saved to '{name}.pkl'")

    print(f"Peak Accuracy: {max(accuracies):.2f}% @ Epoch {accuracies.index(max(accuracies))+1}")
    print(f"Time spent training: {(datetime.now() - start_time).total_seconds():.2f}s")
    return accuracies, losses


def visualize(epochs, acc, loss):
    fig, axs = plt.subplots(ncols=2, figsize=(9, 3), layout="constrained")
    fig.suptitle("Performance")
    ax1, ax2 = axs[0], axs[1]

    epochs = range(1, epochs+1)
    ax1.plot(epochs, acc, "tab:purple")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.grid(True)

    ax2.plot(epochs, loss, "tab:orange")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epochs")
    ax2.grid(True)

    save_path = Path.cwd() / "performances"
    plt.savefig(str(save_path) + f"performance_{int(time.time())}")

