# DrowsyEyeAlert
# Drowsiness Detection using MobileNet

## Overview
This repository contains code for a machine learning model that detects drowsiness based on eye state using the MobileNet architecture. Drowsiness detection is crucial for applications in driver safety and monitoring systems.

The project leverages deep learning techniques to classify whether eyes are open or closed from images captured by a camera. The model is trained on a dataset consisting of images of open and closed eyes.

## MobileNet
MobileNet is a lightweight deep learning model designed for mobile and embedded vision applications. Developed by Google, MobileNet is known for its efficiency and compact size, making it ideal for deployment on devices with limited computational resources.

### Structure
MobileNet utilizes depthwise separable convolutions to reduce the number of parameters and computational cost while maintaining high accuracy. The architecture consists of:
- **Depthwise Separable Convolution:** Splits convolution into a depthwise convolution and a pointwise convolution, reducing computation.
- **Depthwise Convolution:** Applies a single filter per input channel.
- **Pointwise Convolution:** Applies a 1x1 convolution to combine the outputs of the depthwise convolution.

### Advantages of MobileNet
- **Efficiency:** MobileNet achieves high accuracy with fewer parameters, making it suitable for resource-constrained devices.
- **Speed:** Due to its lightweight structure, MobileNet can perform inference quickly.
- **Versatility:** It can be adapted for various tasks such as image classification, object detection, and more.
## Getting Started
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- NumPy
## Results
After training the model on a dataset of open and closed eye images, the model achieved the following performance metrics:

- **Accuracy:** 0.9955
- **Precision:** 0.9885
- **Recall:** 0.9985
- **F1-Score:** 0.9935

### Confusion Matrix

The confusion matrix shows the performance of the model in classifying open and closed eyes:

|             | Predicted Closed Eye | Predicted Open Eye |
|-------------|----------------------|--------------------|
| Actual Closed Eye | 1304                 | 8                  |
| Actual Open Eye   | 1                    | 687                |

This means:
- **True Positive (TP):** 1304 images of closed eyes correctly predicted.
- **False Positive (FP):** 8 images incorrectly predicted as closed eyes.
- **False Negative (FN):** 1 image incorrectly predicted as open eye.
- **True Negative (TN):** 687 images of open eyes correctly predicted.

### Model Performance

The results indicate strong performance; however, with a larger and more diverse dataset, the model's performance could potentially improve further. Due to device limitations, the training data was restricted to 2000 samples.

### Future Scope

The project can be scaled to enable real-time drowsiness detection using live camera feeds and sensors. With advancements in hardware capabilities and access to larger datasets, the model can be enhanced to achieve more robust performance in various real-world scenarios.
