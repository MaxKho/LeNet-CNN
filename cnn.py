# -*- coding: utf-8 -*-

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
train_set = torchvision.datasets.FashionMNIST(root = ".", train=True,
download=True,
transform=transforms.ToTensor())
validation_set = torchvision.datasets.FashionMNIST(root = ".", train=False,
download=True,
transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)
# Fix the seed to be able to get the same randomness across runs and
# hence reproducible outcomes
torch.manual_seed(0)

class myCNN(nn.Module):
    def __init__(self, height=28, width=28, batch_size=32, input_channels=1):
        super().__init__()
        layers = []
        input = input_channels
        for output in [32, 64]:
          layers.append(nn.Conv2d(input, output, (5,5), (1,1)))
          layers.append(nn.ReLU())
          layers.append(nn.MaxPool2d((2,2), (2,2)))
          input = output

        layers.append(nn.Flatten())
        input = 1024
        for output in [1024, 256, 10]:
          layers.append(nn.Linear(input, output))
          layers.append(nn.ReLU()) if output != 10 else None
          input = output

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def train_and_evaluate(self, lr=0.1, epochs=30,
                          initialiser=nn.init.xavier_uniform_,
                          optimiser=torch.optim.SGD,
                          criterion=nn.CrossEntropyLoss(),
                          train_loader=train_loader,
                          test_loader=validation_loader):

        self.apply(lambda m: initialiser(m.weight) if hasattr(m, 'weight') else None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimise = optimiser(self.parameters(), lr=lr)
        self.train_acc, self.val_acc, self.train_loss, self.val_loss = [], [], [], []

        for epoch in range(epochs):
            self.train()
            epoch_loss, correct, total = 0, 0, 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimise.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimise.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

            self.train_acc.append(correct / total)
            self.train_loss.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs} complete. Epoch Training Loss: {epoch_loss:.4f}')
            print(f'Epoch Training Accuracy: {correct / total:.4f} ({correct}/{total} correct)')

            self.eval()
            epoch_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    _, predicted = torch.max(output, dim=1)
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
            self.val_acc.append(correct / total)
            self.val_loss.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs} complete. Epoch Validation Loss: {epoch_loss:.4f}')
            print(f'Epoch Validation Accuracy: {correct / total:.4f} ({correct}/{total} correct)')

        print(f'Training complete.')

model = myCNN()
model.train_and_evaluate()
torch.save(model.state_dict(), "best_cnn.pth")

plt.figure()
plt.plot(model.train_acc, label="Train Accuracy")
plt.plot(model.val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy per Epoch")
plt.show()

plt.figure()
plt.plot(model.train_loss, label="Train Loss")
plt.plot(model.val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss per Epoch")
plt.show()