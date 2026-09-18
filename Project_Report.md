# Project Report: Deepfake and Synthetic Image Detector

## 1. Introduction
With the rapid advancement of Generative AI, the proliferation of hyper-realistic, AI-generated images has become a significant challenge for digital trust. Specifically, platforms such as dating sites and matrimony portals are increasingly targeted by automated "deepfake" profiles. This project aims to build a robust, efficient binary classifier that can distinguish between real human photographs and synthetically generated images.

## 2. Problem Statement
The objective is to develop a backend tool that classifies an uploaded face or image as either "Real" or "Fake" (Synthetic). The tool must be:
- **Accurate**: High F1-score to balance precision and recall.
- **Efficient**: Capable of running on standard server hardware or edge devices.
- **Explainable**: Providing confidence scores to help human moderators make informed decisions.

## 3. Dataset: CIFAKE
The model is trained on the **CIFAKE: Real and AI-Generated Synthetic Images** dataset.
- **Real Images**: Collected from the CIFAR-10 dataset (photographs of natural objects).
- **Synthetic Images**: Generated using specialized Stable Diffusion models to mimic the style and content of the real images.
- **Structure**: The dataset is organized into `train` and `test` directories, each containing `REAL` and `FAKE` subfolders, which simplifies the data loading pipeline using standard PyTorch utilities.

## 4. Methodology and Architecture
### 4.1. Transfer Learning
Instead of training a deep convolutional neural network (CNN) from scratch—which is computationally expensive and prone to overfitting on smaller datasets—we utilized **Transfer Learning**. We adopted **MobileNetV2**, pre-trained on the ImageNet-1K dataset, as our feature extractor.

### 4.2. MobileNetV2 Backbone
MobileNetV2 was chosen for its state-of-the-art efficiency:
- **Depthwise Separable Convolutions**: Significantly reduces the number of parameters and computations.
- **Inverted Residuals**: Optimized for mobile and edge devices.
- **Lightweight**: The final model is under 4MB, ensuring rapid inference times.

### 4.3. Custom Classification Head
The original 1000-class ImageNet head was replaced with a custom binary classifier:
1.  **Linear Layer (1280 → 512)**: Projects high-dimensional features into a dense representation.
2.  **ReLU Activation**: Introduces non-linearity.
3.  **Dropout (0.4)**: A regularization technique to prevent overfitting by randomly muting neurons during training.
4.  **Linear Layer (512 → 2)**: Outputs raw scores (logits) for the "Real" and "Fake" classes.

## 5. Implementation Details
### 5.1. Data Preprocessing and Augmentation
To improve the model's ability to generalize to unseen data, we applied a series of transformations:
- **Training**: Random resizing, cropping (224x224), horizontal flips, color jittering (brightness/contrast), and random rotations.
- **Validation/Testing**: Center cropping and normalization using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 5.2. Training Pipeline
- **Loss Function**: Cross-Entropy Loss, the industry standard for classification.
- **Optimizer**: Adam optimizer with a learning rate of 1e-3 and weight decay for regularization.
- **Scheduler**: `ReduceLROnPlateau` to automatically halve the learning rate if the validation loss stalls.
- **Early Stopping**: Monitored validation loss with a patience of 5 epochs to prevent overfitting and restore the best-performing weights.

### 5.3. Inference Script
A standalone CLI script (`predict.py`) was developed to allow end-users to pass an image path and receive a formatted report. The script applies a Softmax function to the model's output to provide readable confidence percentages.

## 6. Challenges in Detecting AI Artifacts
AI-generated images often contain subtle statistical patterns that are nearly invisible to the human eye but detectable by deep neural networks:
- **Spectral Anomalies**: Unnatural high-frequency patterns caused by the upsampling layers in generative models.
- **Geometric Inconsistencies**: Errors in complex textures like hair, skin pores, or background objects.
- **Color Space Bias**: Synthetic models often have a narrower or shifted color distribution compared to natural photography.

## 7. Conclusion
The developed system provides a reliable and scalable solution for deepfake detection. By combining the efficiency of MobileNetV2 with robust training strategies like transfer learning and data augmentation, we achieved a high-performance tool suitable for integration into real-world web platforms.
