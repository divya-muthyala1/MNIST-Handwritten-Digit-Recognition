import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mnist_model.keras")

## prediction for single digit at index 0

#predictions = model.predict(X_test)
#predicted_digit = np.argmax(predictions[0])
#print("Predicted Digit:", predicted_digit)
#print("Actual Digit   :", y_test[0])

# Load the image
image_name = input("Enter image name: ")

img = Image.open(image_name)

# Convert to grayscale
img = img.convert("L")

# Resize to 28x28
img = img.resize((28, 28))

# Convert image to NumPy array
img_array = np.array(img)

# Display image
plt.imshow(img_array, cmap="gray")
plt.title("Your Image")
plt.axis("off")
plt.show()

# Normalize
img_array = img_array / 255.0

# Reshape
img_array = img_array.reshape(1, 28, 28)

# Predict
prediction = model.predict(img_array)

# Get predicted digit
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)