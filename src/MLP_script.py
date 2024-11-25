# %% [markdown]
# This notebook serializes the method for data preprocessing, training and evaluation of the models. The code below this block Defines the Data preprocessing procedure, the training procedure and evaluation code. All the models utilize the same functions for obtaining the output. Please run all the blocks to successfully traing and evaluate the models. Additionally, for this Model we r
# 
# Data Preprocessing:
# -Changing the dataset to Tensor
# -Normalizing the dataset
# -Importing the dataset using CIFAR10 from torch
# -Using Subset from torch to make subset of the data for 500 images per class for training and 100 for testing
# -Making dataloaders to set the batch size for training and testing
# 
# Training:
# -Training all the models for 100 epochs
# -Cross Entropy Loss function for all models
# -Optmizer is SGD with learning rate 0.01 and momentum of 0.9
# 
# Evaluation:
# -We use sickit learn for calculating the metrics
# -The function evaluate model prints the metrics for all the models
# 

# %%
#libraries imported
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from torchvision.models import resnet18
import time


#Note that the model is trained on cpu
# Define transformations for resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet-18
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the full CIFAR-10 dataset
train_dataset_full = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset_full = CIFAR10(root='./data', train=False, transform=transform, download=True)

# Function to filter the dataset
def filter_dataset(dataset, num_samples_per_class):
    class_counts = {i: 0 for i in range(10)}  # Track counts for each class (0 to 9)/10 classes
    indices = []  # Store selected indices

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < num_samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        # Stop early if all classes have enough samples
        if all(count >= num_samples_per_class for count in class_counts.values()):
            break

    return Subset(dataset, indices)

# Filter datasets
train_dataset = filter_dataset(train_dataset_full, num_samples_per_class=500)
test_dataset = filter_dataset(test_dataset_full, num_samples_per_class=100)
print(train_dataset_full.classes)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# Load pre-trained ResNet-18 and remove the last layer
model = resnet18(pretrained=True)

feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
feature_extractor.eval()  # Set to evaluation mode

# Extract features from the dataset
def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs).squeeze()
            features.append(outputs)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

# Extract features for training and testing
train_features, train_labels = extract_features(train_loader, feature_extractor)
test_features, test_labels = extract_features(test_loader, feature_extractor)

# %%
#Transform the features to vectors using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features.numpy())
test_features_pca = pca.transform(test_features.numpy())


# %% [markdown]
# The Code block below is for functions: training and evaluation. Train_model performs the training of all the models, it takes parameters as model, train features, train labels, epochs. Although train features and labels don't change, this was done to experiment with the dataset. The evaluation function is also defined in this cell. These are the functions used by the models

# %%


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
criterion = nn.CrossEntropyLoss()


# Training loop
def train_model(model, train_features, train_labels, epochs=100):
    #setting the optimizer, note that that mlp_model is set in the Model definitions when it is called for training
    #It is the same as model taken from the parameter
    optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):

        optimizer.zero_grad()
        outputs = model(torch.tensor(train_features, dtype=torch.float32))

        loss = criterion(outputs, torch.tensor(train_labels, dtype=torch.long))
        loss.backward()

        optimizer.step()
        #un-comment the code below to check loss at epochs
        #print(f'Epoch {epoch+1}, Loss: {loss.item()}')



# Evaluation
def evaluate_model(model, test_features, test_labels):
    model.eval()
    with torch.no_grad():
        
        predictions = model(torch.tensor(test_features, dtype=torch.float32))
        
        predicted_labels = predictions.argmax(dim=1).numpy()
        
        true_labels = test_labels.numpy()

        test_loss = criterion(predictions, test_labels.clone().detach().long()).item()
        
    # Metrics
    acc = accuracy_score(true_labels, predicted_labels)
    
    report = classification_report(true_labels, predicted_labels, target_names=train_dataset_full.classes)

    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))

    #Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset_full.classes, yticklabels=train_dataset_full.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print(f'Accuracy: {acc}')
    print(f'Test Loss: {test_loss}')
    print('Classification Report:\n', report)


# %% [markdown]
# MLP Base Case

# %%

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 512),       # Input layer
            nn.ReLU(),
            nn.Linear(512, 512),      # Hidden layer with batch normalization
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)        # Output layer
        )

    def forward(self, x):
        return self.layers(x)
mlp_model = MLP()
start_time = time.time()
train_model(mlp_model, train_features_pca, train_labels, epochs=100)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")
#This code below is to save the model
#torch.save(model.state_dict(), r'C:\Users\umara\OneDrive\Desktop\University\COMP 472\Project\Results\Models\MLP.pth')

evaluate_model(mlp_model, test_features_pca, test_labels)


# %% [markdown]
# MLP with extra hidden layers

# %%
class MLP_V1(nn.Module):
    def __init__(self):
        super(MLP_V1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Additional hidden layer
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Additional hidden layer
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)    # Output layer
        )

    def forward(self, x):
        return self.layers(x)
    
mlp_model = MLP_V1()

start_time = time.time()
train_model(mlp_model, train_features_pca, train_labels, epochs=100)
end_time = time.time()

#This code below is to save the model
#torch.save(model.state_dict(), r'C:\Users\umara\OneDrive\Desktop\University\COMP 472\Project\Results\Models\MLP_V1.pth')

print(f"Training time: {end_time - start_time} seconds")

evaluate_model(mlp_model, test_features_pca, test_labels)


# %% [markdown]
# MLP with less layers than the base case

# %%
class MLP_V2(nn.Module):
    def __init__(self):
        super(MLP_V2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)
mlp_model = MLP_V2()
start_time = time.time()
train_model(mlp_model, train_features_pca, train_labels, epochs=100)
end_time = time.time()

#This code below is to save the model
#torch.save(model.state_dict(), r'C:\Users\umara\OneDrive\Desktop\University\COMP 472\Project\Results\Models\MLP_V2.pth')

print(f"Training time: {end_time - start_time} seconds")
evaluate_model(mlp_model, test_features_pca, test_labels)

# %% [markdown]
# MLP with Less Neurons in the layer(weights)

# %%
class MLP_V3(nn.Module):
    def __init__(self):
        super(MLP_V3, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
mlp_model = MLP_V3()
start_time = time.time()
train_model(mlp_model, train_features_pca, train_labels, epochs=100)
end_time = time.time()

#This code below is to save the model
#torch.save(model.state_dict(), r'C:\Users\umara\OneDrive\Desktop\University\COMP 472\Project\Results\Models\MLP_V3.pth')

print(f"Training time: {end_time - start_time} seconds")
evaluate_model(mlp_model, test_features_pca, test_labels)

# %% [markdown]
# MLP with more Neurons in the layers(weights)

# %%
class MLP_V4(nn.Module):
    def __init__(self):
        super(MLP_V4, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.layers(x)
mlp_model = MLP_V4()
start_time = time.time()
train_model(mlp_model, train_features_pca, train_labels, epochs=100)
end_time = time.time()

#This code below is to save the model
#torch.save(model.state_dict(), r'C:\Users\umara\OneDrive\Desktop\University\COMP 472\Project\Results\Models\MLP_V4.pth')

print(f"Training time: {end_time - start_time} seconds")
evaluate_model(mlp_model, test_features_pca, test_labels)


