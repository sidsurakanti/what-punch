from torch import nn


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
