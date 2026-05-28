import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Dataset load karo
dataset = pd.read_csv('calories.csv')

print(dataset.head())

# 2. Features aur Target alag karo
X = dataset.drop('Calories', axis=1)
y = dataset['Calories']

# 3. Data ko train aur test mein split karo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Data ko scale karo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. ANN Model banao
model = Sequential()
model.add(Dense(units=12, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1))

# 6. Model compile karo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Model train karo
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, verbose=1)

# 8. Model evaluate karo
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test MAE: {mae}')

# 9. Graph banao
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
