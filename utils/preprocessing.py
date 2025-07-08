import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

#Import data file.

df = pd.read_csv(r"C:\Users\hp\My Project\Project Cat Breeds v2 (Classification) (10 class)\data\cat_breeds_clean.csv")

#Rearrange the rows randomly.

df = df.sample(frac=1).reset_index(drop=True)

#Delete the Origin column, it is not needed.

df.drop('Origin', axis=1, inplace=True)

#Delete rows that contain duplicate data.

df.drop_duplicates(keep='first', inplace=True)

# Convert label data to numeric values (int)

df['Breeds'] = df['Breeds'].astype('category').cat.codes.astype('int64')

# Convert string values to numeric values (One-Hot Encoding)

df = pd.get_dummies(df)

original_columns = df.columns.tolist()

joblib.dump(original_columns, 'dummies_columns.pkl')

bool_columns = df.select_dtypes(include='bool').columns

df[bool_columns] = df[bool_columns].astype('int64')

# Convert values to the range [0, 1] (Normalization)

columns_to_normalize = ['Weight', 'length', 'Age', 'Weight', 'Sleep_hours']

scaler = MinMaxScaler()

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

joblib.dump(scaler, 'scaler.pkl')

# Ensure that there are no object values and that all values are numeric.

print(df.dtypes)

#Download the data to start training the model on it.

df.to_csv('cat_breeds_dataset.csv', index=False)