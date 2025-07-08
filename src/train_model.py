import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read data file (csv)

df = pd.read_csv(r"C:\Users\hp\My Project\Project Cat Breeds v2 (Classification) (10 class)\data\cat_breeds_dataset.csv",delimiter=',')

# Define the dataset class

class CatBreedDataset(Dataset):
  def __init__(self, Data , Labels):
    self.X = torch.tensor(Data, dtype=torch.float32)
    self.y = torch.tensor(Labels, dtype=torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
# Store features in X and labels in y
# Split data into training data and test data
# Define the datasets and dataloaders

X = np.array(df.iloc[:,1:])
y = np.array(df.iloc[:,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

trianDataset = CatBreedDataset(X_train, y_train)
testDataset = CatBreedDataset(X_test, y_test)
valDataset = CatBreedDataset(X_val, y_val)

trainLoader = DataLoader(trianDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=32, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=32, shuffle=False)

# Define the model structure class

class CatClassifier(nn.Module):
  def __init__(self, INPUTS_SIZE, HIDDEN_SIZE_1 = 64, HIDDEN_SIZE_2 = 32, OUTPUTS_SIZE = 10):
    super(CatClassifier, self).__init__()
    self.input_layer = nn.Linear(INPUTS_SIZE, HIDDEN_SIZE_1)
    self.dropout = nn.Dropout(0.3)
    self.hidden_layer = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
    self.dropout = nn.Dropout(0.3)
    self.output_layer = nn.Linear(HIDDEN_SIZE_2, OUTPUTS_SIZE)

  def forward(self, x):
    x = F.relu(self.input_layer(x))
    x = self.dropout(x)
    x = F.relu(self.hidden_layer(x))
    x = self.dropout(x)
    x = self.output_layer(x)
    return x
  
# Create instance from the model
# Define the optimizer and loss function

input_size = X_train.shape[1]

model = CatClassifier(input_size)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Testing the model

def evaluate(model, dataloader, criterion):

  test_loss = 0.0
  correct = 0
  total = 0

  model.eval()
  with torch.no_grad():
    for inputs, targets in dataloader:
      outputs = model(inputs)

      loss = criterion(outputs, targets)
      test_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1)

      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  avg_loss = test_loss / len(dataloader)
  accuracy = 100 * (correct / total)

  return avg_loss, accuracy

# Training loop
# Validating the model

epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
  model.train()
  running_loss = 0.0

  for inputs, targets in trainLoader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  train_loss = running_loss / len(trainLoader)

  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
    for val_inputs, val_targets in valLoader:
      val_outputs = model(val_inputs)
      val_loss = criterion(val_outputs, val_targets)
      _, predicted = torch.max(val_outputs, 1)
      correct += (predicted == val_targets).sum().item()
      total += val_targets.size(0)

  val_accuracy = 100 * (correct / total)

  if epoch % 10 == 0:
    test_loss, test_accuracy = evaluate(model, testLoader, criterion)

  train_losses.append(loss.item())
  val_losses.append(val_loss.item())

  print(f"Epoch {epoch + 1}:- Train Loss: {train_loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, Validation Accuracy: {round(val_accuracy, 2)}, Test Accuracy: {round(test_accuracy, 2)}")

all_labels = []
all_preds = []
model.eval()
with torch.no_grad():
    for inputs, targets in testLoader:
       outputs = model(inputs)
       _, predicted = torch.max(outputs.data, 1)

       all_preds.extend(predicted.cpu().numpy())
       all_labels.extend(targets.cpu().numpy())

#Graph Training vs Validation Loss

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.plot(moving_average(train_losses), label='Train Loss (Smoothed)')
plt.plot(moving_average(val_losses), label='Val Loss (Smoothed)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Smoothed Training vs Validation Loss')
plt.show()

#Calculation of the confusion matrix, classification report and accuracy score.

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrics")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

class_name = ['Abyssinian','British Shorthair','Egyption Mau', 'Japanese Bobtail', 'Maine Coon', 'Manx',
              'Norwegian Forest Cat', 'Persian', 'Siamese', 'Turkish Angora']

print(classification_report(all_labels, all_preds, target_names = class_name))

acc = accuracy_score(all_labels, all_preds)

print(f"Accuracy: {acc * 100}%")

#Keep model weights

torch.save(model.state_dict(), "best_model.pt")