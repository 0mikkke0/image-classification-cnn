# Image Classification with CNN

An end-to-end deep learning project that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN).

## What It Demonstrates

* Computer vision fundamentals
* Convolutional Neural Network (CNN) architecture
* TensorFlow / Keras
* Image preprocessing and normalization
* Training and validation
* Data augmentation
* Model evaluation
* Confusion matrix and classification report
* Saving and loading a trained model
* Streamlit image-upload prediction interface

## Dataset

CIFAR-10 is automatically downloaded by TensorFlow/Keras. It contains 60,000 32×32 color images across 10 classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

No dataset needs to be uploaded manually to GitHub.

## Project Structure

```text
image-classification-cnn/
├── data/
├── models/
├── notebooks/
│   └── cnn_cifar10_analysis.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Recommended: Python 3.11 or 3.12.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate.bat
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the CNN:

```bash
python src/train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

The first training run automatically downloads the CIFAR-10 dataset through TensorFlow/Keras.

## Model

The CNN uses convolutional layers to learn spatial patterns from images, pooling layers to reduce spatial dimensions, and dropout for regularization.

The final softmax layer produces probabilities across the 10 CIFAR-10 classes.

Data augmentation is applied during training to improve model generalization and reduce overfitting.

## Evaluation

The training script evaluates the model using:

* Test accuracy
* Classification report
* Confusion matrix

The trained model is saved to:

```text
models/cifar10_cnn.keras
```

## Results

Results will be added after training and evaluation.

Example:

```text
Test Accuracy: XX.XX%
```

The model is also evaluated using a classification report and confusion matrix across all 10 CIFAR-10 classes.

## Key Deep Learning Concepts

This project demonstrates practical understanding of:

* Convolutional Neural Networks (CNNs)
* Convolution and learned filters
* Pooling layers
* ReLU activation
* Softmax classification
* Categorical cross-entropy
* Epochs and batch size
* Dropout regularization
* Data augmentation
* Image normalization
* Train/validation/test splitting
* Confusion matrices
* Overfitting and generalization
* Transfer learning as a potential improvement

## Future Improvements

Possible improvements include:

* Transfer learning using pretrained architectures such as ResNet or MobileNet
* Hyperparameter tuning
* Learning-rate scheduling
* More extensive data augmentation
* Model performance analysis by individual CIFAR-10 class
* Deployment using a production-oriented serving framework

## Disclaimer

This is a portfolio/learning project using the public CIFAR-10 benchmark. It is not intended to represent a production computer-vision system.
