import torch
from torch import nn
from torch.utils.data import DataLoader
from Datahandler import *
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#training_data = CustomDatasetTest()
#test_data = CustomDatasetTest()
training_data = CustomDataset("Chrom63train.csv")
test_data = CustomDataset("Chrom63test.csv")
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(len(train_dataloader.dataset))
print(len(train_dataloader))
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9,7),
            nn.ReLU(),
            nn.Linear(7, 5),
            nn.ReLU(),
            nn.Linear(5, 2)

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print("batch:", batch, X[0])
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        #print(pred,loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_loss2 = 0, 0
    with torch.no_grad():
        first = True
        for X, y in dataloader:
            pred = model(X)
            pred2 = torch.round(3000*pred)
            if first:
                first = False
                print("pred:", pred2[0])
                print("actual:", 3000*y[0])
            
            test_loss += loss_fn(pred, y).item()
            test_loss2 += loss_fn(pred2, 3000*y).item()
    test_loss /= num_batches
    test_loss2 /= num_batches
    print(f"Test Error: \n Avg loss1: {test_loss:>8f} Avg loss2: {test_loss2:>8f} \n")

epochs = 500

def plotdata(dataloader, model):
    A = 
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            A = torch.round(3000*pred)
            B = 3000*y
            plt.plot(A[:, 0]-A[:, 1], "g", B[:, 0]-B[:, 1], "r")
            plt.xlabel("index")
            plt.legend(["real coeffs", "predicted coeffs(outsample)"])
            plt.show()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")