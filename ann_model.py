# ============================================
# ANN MODEL FOR NUMERICAL/TABULAR DATA
# Google Colab Code
# ============================================

# ============================================
# STEP 1 — IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ============================================
# STEP 2 — LOAD DATASET
# ============================================

# Upload CSV file in Colab first

dataset = pd.read_csv('calories')

# Show first 5 rows
print(dataset.head())

# ============================================
# STEP 3 — INPUT & OUTPUT
# ============================================

# Last column = target/output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ============================================
# STEP 4 — ENCODE OUTPUT LABELS
# ============================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ============================================
# STEP 5 — FEATURE SCALING
# ============================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================================
# STEP 6 — TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# STEP 7 — BUILD ANN MODEL
# ============================================

model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(
    units=64,
    activation='relu',
    input_dim=X_train.shape[1]
))

model.add(Dropout(0.2))

# Hidden Layer 2
model.add(Dense(
    units=32,
    activation='relu'
))

model.add(Dropout(0.2))

# Output Layer
model.add(Dense(
    units=len(np.unique(y)),
    activation='softmax'
))

# ============================================
# STEP 8 — COMPILE MODEL
# ============================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# STEP 9 — TRAIN MODEL
# ============================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# ============================================
# STEP 10 — EVALUATE MODEL
# ============================================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# ============================================
# STEP 11 — SAVE MODEL
# ============================================

model.save("ann_model.h5")

print("Model Saved Successfully!")

# ============================================
# STEP 12 — PREDICTION
# ============================================

sample = X_test[0]

sample = np.expand_dims(sample, axis=0)

prediction = model.predict(sample)

predicted_class = np.argmax(prediction)

print("Predicted Class:",
      label_encoder.inverse_transform([predicted_class]))

# ============================================
# STEP 13 — ACCURACY GRAPH
# ============================================

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()
