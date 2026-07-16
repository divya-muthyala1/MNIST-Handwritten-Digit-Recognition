import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

# Loading the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# checking the shape of the data
print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images  :", X_test.shape)
print("Testing Labels  :", y_test.shape)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Creating ANN
model = Sequential()

model.add(Input(shape=(28,28)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()
# Until now, we have built the neural network.
#But the model still doesn't know how to learn.

# Compile the model, this does not train the model but it prepares the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] 
    )

# Trains the model
history = model.fit(
    X_train, y_train, epochs=5, batch_size=32, validation_split=0.2
    )

# Evaluate the model
model.evaluate(X_test, y_test)

## Training Accuracy
plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

## Training Vs Validation Accuracy
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

# Save the model
model.save("mnist_model.keras")