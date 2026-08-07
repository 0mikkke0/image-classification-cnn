# Image Classification with CNN

An end-to-end deep learning project that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN).

## What it demonstrates
- Computer vision fundamentals
- CNN architecture
- TensorFlow / Keras
- Image preprocessing and normalization
- Training / validation split
- Data augmentation
- Model evaluation
- Confusion matrix and classification report
- Saving and loading a trained model
- Streamlit image-upload prediction UI

## Dataset
CIFAR-10 is automatically downloaded by TensorFlow/Keras. It contains 60,000 32x32 colour images across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

No dataset needs to be uploaded manually to GitHub.

## Structure
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

```bash
python -m venv .venv
```

Windows CMD:
```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The first training run downloads CIFAR-10 automatically.

## Model
The CNN uses convolutional layers to learn spatial patterns, pooling layers to reduce spatial dimensions, dropout for regularization, and a softmax output for the 10 classes.

Data augmentation is used during training to improve generalization.

## Evaluation
The training script reports:
- test accuracy
- classification report
- confusion matrix

The saved model is written to `models/cifar10_cnn.keras`.

## Interview topics
Be ready to explain:
- why CNNs are useful for images,
- convolution and filters,
- pooling,
- ReLU,
- softmax,
- categorical cross-entropy,
- epochs and batch size,
- overfitting,
- dropout,
- data augmentation,
- train/validation/test split,
- why normalization helps,
- confusion matrices,
- transfer learning and how you could improve this project.

## Disclaimer
This is a portfolio/learning project using the public CIFAR-10 benchmark. It is not a production computer-vision system.
