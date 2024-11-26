import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import copy
# transform
transform = transforms.Compose([
    transforms.ToTensor(),  # picture to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalization
])

# download
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# divide
num_local_devices = 5
divide_list = []
train_size_tot = 0
for i in range(num_local_devices):
    local_train_size = int(0.8 * len(train_set) / 20 )
    divide_list.append(local_train_size)
    train_size_tot += local_train_size

central_train_size = len(train_set) - train_size_tot

divide_list.append(central_train_size)

dataset_list =  random_split(train_set, divide_list)
print("length: ",len(dataset_list))
#train_dataset, val_dataset = random_split(train_set, [train_size, central_train_size])

#print(random_split(train_set, [train_size, central_train_size]))
# data loader

# central_train_size = DataLoader(val_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)#log_

def train_model(model, train_loader, optimizer, criterion, epochs=2):#, val_loader
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        train_losses.append(train_loss)
    return model



local_models = []
for i in range(num_local_devices):
    train_dataset = dataset_list[i]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("device: ", i)
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = train_model(model, train_loader, optimizer, criterion, epochs=10)#, val_loader
    one_local_model = copy.deepcopy(model)
    local_models.append(one_local_model)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    #precision = 0
    print(f'Test Accuracy: {accuracy:.4f}')
    #print(f'Test Precision: {precision:.4f}')

evaluate_model(local_models[0], test_loader)

central_train_dataset = dataset_list[-1]
central_train_loader = DataLoader(central_train_dataset, batch_size=64, shuffle=True)

def evaluate_avg_model(local_models, test_loader):
    
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in test_loader:
            outputs = 0
            for i in range(len(local_models)):
                model = local_models[i]
                model.eval()
                outputs += model(images)
            outputs = outputs/len(local_models)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    #precision = 0
    print(f'Avg Test Accuracy: {accuracy:.4f}')

evaluate_avg_model(local_models, test_loader)

# for i in range(num_local_devices):
#     print("evaluate device: ", i)
#     evaluate_model(local_models[i], test_loader)

def train_central_model(model, local_models, train_loader, optimizer, criterion, epochs=2):#, val_loader
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            tot_evaluate_results = 0
            #print("check")
            for i in range(len(local_models)):
                model = local_models[i]
                model.eval()
                tot_evaluate_results += model(images)#.item()
            teacher_labels = tot_evaluate_results / len(local_models)
            #print("teacher_labels:",teacher_labels)
            loss = criterion(outputs, teacher_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        train_losses.append(train_loss)
    return model

central_model = SimpleCNN()
central_optimizer = optim.SGD(central_model.parameters(), lr=0.001, momentum=0.9)#lr should be one order of magnitude smaller than before!
central_criterion = nn.CrossEntropyLoss()
trained_central_model = train_central_model(central_model, local_models, central_train_loader, optimizer, criterion, epochs=2)
evaluate_model(trained_central_model, test_loader)