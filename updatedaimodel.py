import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os

# Safe = 50%, Dangerous = 50%
def create():
    np.random.seed(42)
    drugs = [f"Drug_{i}" for i in range(1, 101)]
    unique_pairs = [(d1, d2) for i, d1 in enumerate(drugs) for d2 in drugs[i + 1:]]
    interactions = np.random.choice([0, 1], size=len(unique_pairs), p=[0.5, 0.5])
    df = pd.DataFrame(unique_pairs, columns=['Drug A', 'Drug B'])
    df['Interaction'] = interactions
    df.to_csv('drug_interaction_dataset.csv', index=False)

#Load Dataset
if not os.path.exists('drug_interaction_dataset.csv'):
    print("Dataset not found ")
    create()
else:
    print("CSV already exists ")

df = pd.read_csv('drug_interaction_dataset.csv')

# Encode drug names
encoder = LabelEncoder()
df['Drug A'] = encoder.fit_transform(df['Drug A'])
df['Drug B'] = encoder.fit_transform(df['Drug B'])

#Features and Labels
X = df[['Drug A', 'Drug B']].values
y = df['Interaction'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Custom Dataset Class
class DrugDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

#DataLoader
train_dataset = DrugDataset(X_train, y_train)
test_dataset = DrugDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Model Architecture
class DrugInteractionModel(nn.Module):
    def __init__(self):
        super(DrugInteractionModel, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = DrugInteractionModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

#Example Prediction
# Training
print("\nTraining Model \n")
for epoch in range(200):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Training Cycle {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features).squeeze()
        predictions = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n Model Accuracy: {accuracy:.2f}%\n")

encoded_drug_a = encoder.transform(['Drug_1'])[0]
encoded_drug_b = encoder.transform(['Drug_4'])[0]
new_input = torch.tensor([encoded_drug_a, encoded_drug_b], dtype=torch.float32)
model.eval()

with torch.no_grad():
    output = model(new_input)
    prediction = torch.round(output).item()

if prediction == 1:
    print("Dangerous Combination Detected")
else:
    print("Safe Combination")
