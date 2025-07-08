import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import gradio as gr

# Structure of the trained model.

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
  
# A function to process new data and enter it into a model to predict the label.
  
def prediction(Weight, length, Fur_length, Fur_type, Fur_color, Eye_color, Age, Sleep_hours):
    model = CatClassifier(34)
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    original_columns = joblib.load('artifacts/dummies_columns.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    new_data = pd.DataFrame({
             'Weight' : [Weight],
             'length' : [length],
             'Fur_length' : [Fur_length],
             'Fur_type' : [Fur_type],
             'Fur_color' : [Fur_color],
             'Eye_color' : [Fur_color],
             'Age' : [Age],
             'Sleep_hours' : [Sleep_hours],
                                           })
    new_data_encoded = pd.get_dummies(new_data)
    new_data_encoded = new_data_encoded.reindex(columns=original_columns, fill_value=0)
    bool_columns = new_data_encoded.select_dtypes(include='bool').columns
    new_data_encoded[bool_columns] = new_data_encoded[bool_columns].astype('int64')
    columns_to_normalize = ['Weight', 'length', 'Age', 'Weight', 'Sleep_hours']
    new_data_encoded[columns_to_normalize] = scaler.transform(new_data_encoded[columns_to_normalize])
    new_data_encoded = new_data_encoded.drop('Breeds', axis=1)
    x = torch.tensor(new_data_encoded.values, dtype=torch.float32)
    with torch.no_grad():
       output = model(x)
       _, predicted = torch.max(output, 1)
       breed_map = {0: 'Abyssinian', 1: 'British Shorthair', 2: 'Egyption Mau', 3: 'Japanese Bobtail', 4: 'Maine Coon', 5: 'Manx',
                    6: 'Norwegian Forest Cat', 7: 'Persian', 8: 'Siamese', 9: 'Turkish Angora'}
       predicted_breed = breed_map.get(predicted.item())
       return predicted_breed
    
# Create a user interface to receive new data and output label (Gradio).

def launch_app (): 
  app = gr.Interface(
      fn = prediction,
      inputs = [
          gr.Textbox(label = "Weight"),
          gr.Textbox(label = "Length"),
          gr.Dropdown(["long", "short", "short/medium", "medium/long"], label = "Fur Length"),
          gr.Dropdown(["soft", "silky", "heavy", "Thick/soft"], label = "Fur Type"),
          gr.Dropdown(["white", "black", "gray", "silver", "brown", "red", "ruddy", "blue", "orange", "smokey", "creamy", "calcalico",
                       "bronze", "Bluish grey"], label = "Fur Color"),
          gr.Dropdown(["Hazelnut", "amber", "blue", "copper", "golden", "green", "pink", "two different eyes"], label = "Eye Color"),
          gr.Textbox(label = "Age"),
          gr.Textbox(label = "Sleep Hours"),
                                            ],
      outputs = "text",
      title = "Prediction Cat Breeds",
      description = "Fill in the following information so I can predict your cat's breed."
      )
  
  app.launch(share = False, inbrowser = True) # Run the user interface.
