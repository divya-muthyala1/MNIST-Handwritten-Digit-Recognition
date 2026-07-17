# MNIST Handwritten Digit Recognition using ANN

##  Project Overview

This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. The trained model can classify digits from the dataset as well as predict custom handwritten digit images.


##  Features

- Load the MNIST dataset
- Preprocess image data
- Build an ANN using TensorFlow/Keras
- Train the neural network
- Evaluate model performance
- Save the trained model
- Load the saved model
- Predict custom handwritten digit images


##  Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow

##  Project Structure

MNIST-Handwritten-Digit-Recognition/
│
──>train_model.py
──> predict_digit.py
──>mnist_model.keras
──>digit.png
──>requirements.txt
──>README.md
──>.gitignore
──>screenshots/
──>venv/

##  How to Run

### 1. Clone the repository

```bash
git clone <repository-link>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Predict your own handwritten digit

Replace `digit.png` with your own handwritten digit image and run:

```bash
python predict_digit.py
```

---

##  Model Architecture

- Input Layer (28 × 28)
- Flatten Layer
- Dense Layer (128 neurons, ReLU)
- Output Layer (10 neurons, Softmax)

---

##  Results

- Successfully classified handwritten digits from the MNIST dataset.
- Successfully predicted custom handwritten digit images.

---

##  Screenshots

Screenshots of the training graphs and prediction results can be found in the `screenshots` folder.

---

## Note: 

- This model is trained using the MNIST handwritten digit dataset. While it performs well on the official MNIST test set.
- predictions on custom handwritten images may vary because such images often differ in size, position, stroke thickness, and background. 
- For reliable predictions on custom images, additional preprocessing (cropping, centering, and normalization) should be applied before inference.

