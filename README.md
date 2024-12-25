# Deep Fake Detection and Classification

## 1. Overview

Deep Fake Detection and Classification is a project aimed at identifying and classifying images as either real or fake. This system leverages deep learning techniques to achieve high accuracy in distinguishing between real and deep fake images. The dataset used for this project contains a diverse collection of images sourced from multiple directories, including manually labeled real and fake images.

### Dataset

The dataset utilized for training and evaluation is a combination of:

- **140k Real and Fake Faces**
- **Deep Fake and Real Images**
- **Hard Fake vs Real Faces**
- **Real and Fake Face Detection**
- **Real vs AI-Generated Faces Dataset**

<img src="https://github.com/leovidith/DeepFake-Image-Classifier/blob/main/curves.png" width=700px>

#### Dataset Link:

[Deepfake and Real Images on Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

The dataset was preprocessed and consolidated into a CSV file containing image paths and corresponding labels ("Real" or "Fake"). Images were resized to a target size of 150x150 pixels for input into the model.

## 2. Results

The deep learning model achieved the following results after 25 epochs:

- **Training Loss**: 0.1060
- **Training Accuracy**: 96.02%
- **Validation Loss**: 0.2002
- **Validation Accuracy**: 93.30%

### Performance Graphs

<img src="https://github.com/leovidith/DeepFake-Image-Classifier/blob/main/accuracy%20curves.png" width=500px>
<img src="https://github.com/leovidith/DeepFake-Image-Classifier/blob/main/loss%20curves.png" width=500px>

#### Loss:

- Training Loss: Reduced steadily over epochs.
- Validation Loss: Slightly higher than training loss but remained stable.

#### Accuracy:

- Training Accuracy: Increased consistently, reaching 96%.
- Validation Accuracy: Achieved 93.3%, indicating good generalization.

## 3. Agile Features

The project was developed using an agile methodology, divided into two main sprints:

### Sprint 1: Data Preparation and Preprocessing

- Consolidated multiple datasets into a single CSV file.
- Applied class mapping to unify label formats (e.g., "Fake", "Real").
- Resized images to 150x150 pixels and implemented data augmentation using `ImageDataGenerator`.
- Split data into training (80%) and validation (20%) subsets.

### Sprint 2: Model Development and Training

- Built a Convolutional Neural Network (CNN) using TensorFlow and Keras.
- Configured layers with increasing filter sizes (32, 64, 128, 256) and dropout for regularization.
- Trained the model for 25 epochs with a batch size of 32.
- Visualized training progress using loss and accuracy curves.

## 4. Conclusion

The project successfully developed a robust deep learning model to detect and classify deep fake images. With an accuracy of 93.3% on the validation set, the model demonstrates strong potential for real-world applications in combating misinformation and identifying synthetic media. Future improvements could include:

- Expanding the dataset with more diverse examples.
- Experimenting with advanced architectures like EfficientNet or ResNet.
- Fine-tuning hyperparameters for further optimization.

This project serves as a foundational step toward automated deep fake detection and can be extended for video-based deep fake analysis.

