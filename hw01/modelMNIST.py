from torch import nn

import loadMNISTData
import settingDevice

class ClassificationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10), # 10 class
            # not insert softmax using CrossEntropyLoss
        )

    def foward(self, x):
        x = x.view(x.size(0), -1) # (batch, 1, 28, 28) -> (batch, 784)
        return self.model(x)

model = ClassificationMLP().to(settingDevice.device)


if __name__ == "__main__":
    ClassificationMLP()
    print(model)