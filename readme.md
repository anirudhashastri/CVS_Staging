# Enhancing Bone Age Prediction through Deep Learning: A Comparative Study of U-Net and InceptionResNetV2

## Project Overview

This research project aims to enhance bone age prediction using deep learning techniques with a focus on cervical vertebrae staging (CVS). By employing two models, U-Net and InceptionResNetV2, this study seeks to determine which model offers superior accuracy and efficiency in the context of CVS staging. The project's primary goal is to improve safety and reduce radiographic exposure in bone age prediction by finding the best approach for CVS staging.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Models and Implementation](#models-and-implementation)
   - [Model Descriptions](#model-descriptions)
   - [Implementation](#implementation)
4. [Evaluation and Results](#evaluation-and-results)
5. [How to Run the Repository](#how-to-run-the-repository)
6. [Conclusion](#conclusion)

## Introduction

Bone age determination is a crucial aspect of assessing skeletal maturation. Traditionally, the Cervical Vertebrae Stage (CVS) method involves manual tracing of lateral cephalograms to classify age stages based on the vertebrae's shape and curvature. This project utilizes deep learning to automate this process, thus minimizing human error and speeding up the analysis.

## Dataset

The dataset consists of 400 lateral cephalogram images, curated by me through a collaboration between my department and the university's dental college. These images are annotated with 19 specific points on the cervical vertebrae, with coordinates stored in a corresponding CSV [Train](https://github.com/anirudhashastri/CVS_Staging/blob/main/Data/train_data.csv) [Test](https://github.com/anirudhashastri/CVS_Staging/blob/main/Data/test_data.csv) file as can be seen in the [Annotated Data](https://github.com/anirudhashastri/CVS_Staging/tree/main/Data/Annotated-Data)  . Data preprocessing involved extracting the vertebral column from each image and enhancing image contrast using Adaptive Histogram Equalization.

## Models and Implementation

### Model Descriptions

#### InceptionResNetV2

- **Feature Extraction**: Pre-trained on ImageNet, this model's layers remain non-trainable to preserve learned weights.
- **Architecture**: Includes a Flatten layer, followed by Dense layers with ReLU activation, and an output layer with a linear activation function.
- **Output**: Designed for regression or multi-class classification of 38 different classes.

![Unet-predicted side by side (1)](https://github.com/anirudhashastri/CVS_Staging/blob/main/Results/Inception-prediction.png))

#### U-Net

- **Architecture**: Symmetrical structure with an encoder (contracting path) and a decoder (expanding path) connected by skip connections.
- **Encoder**: Multiple convolutional blocks with ReLU activation, batch normalization, dropout for regularization, and max pooling.
- **Decoder**: Upsampling blocks that increase spatial resolution, incorporating skip connections from the encoder.
- **Output Layer**: Concludes with a convolutional layer using a sigmoid activation function for pixel-wise classification.
![unet-Overlap (1)](https://github.com/anirudhashastri/CVS_Staging/blob/main/Results/unet-Overlap.png)
### Implementation

- **Software and Tools**: TensorFlow 2.10, Keras, and Python 3.10. Training scripts executed in Jupyter Notebooks on a Windows system.
- **Hardware**: Training conducted using an NVIDIA RTX 3090 GPU.
- **Training Parameters**: U-Net trained for 1000 epochs, InceptionResNetV2 for 600 epochs. Adam optimizer with learning rates of 0.001 for U-Net and 0.01 for InceptionResNetV2. Batch size set at 32.
- **Loss Functions**: Dice Loss for U-Net, Mean Absolute Error (MAE) for InceptionResNetV2.

## Evaluation and Results

- **Metrics**: Percentage of Correct Keypoints (PCK) for accuracy in landmark detection. U-Net achieved 85% PCK accuracy and a Dice Loss of 0.25. InceptionResNetV2 achieved 75% PCK accuracy and an MAE of 15.2.
- **Findings**: U-Net demonstrated solid performance in general mask localization, while InceptionResNetV2, though underperforming slightly due to dataset constraints, is expected to improve significantly with a larger dataset.

| Model            | Accuracy (PCK) | MAE  | Dice Loss |
|------------------|----------------|------|-----------|
| U-Net            | 85%            | -    | 0.25      |
| InceptionResNetV2| 75%            | 15.2 | -         |

## How to Run the Repository

To run this project, follow these steps:

### 1. Pull the Repository

Clone the repository to your local machine using the following command:

######   
      git clone https://github.com/anirudhashastri/cvs-staging.git

### 2. Packages Required
Download and install [anaconda](https://www.anaconda.com/download)
Install the required packages using pip:

###### Create the environment
    conda env create -f cvs-staging.yaml
or
Go to the anaconda application and upload the .yml file in your environments tab

### 3. Link to the Pretrained Model
Download the [pretrained Models](https://drive.google.com/drive/folders/1htkTWYttLfAWn--kAW2v6PRjgRgF9OPp?usp=sharing).

### 4. Change File Paths and Model Paths in the Code
Update the file paths and model paths in the code to match your local setup. Open the script files and modify the paths where necessary.

### 5. Sample Data
The repository includes some sample data for testing purposes. However, due to the confidentiality of healthcare data, the full dataset is not provided. You can run this code with any similar dataset by following the same preprocessing steps.

### 6. Running
In vs code open any of the files and for python the anacoda enviromnet we created to run the whole file.

## Conclusion

The U-Net model excels in general mask localization for image segmentation, while InceptionResNetV2 shows potential for precise localization but requires a larger dataset for improved performance. Future work should focus on expanding the dataset to enhance the InceptionResNetV2 model's precision and robustness.



