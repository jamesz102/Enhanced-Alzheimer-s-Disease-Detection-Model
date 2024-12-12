# Disease Recognition: Alzheimer's Disease Detection Project

## Overview
This project implements a hybrid model combining **ResNet50** and **DenseBlocks** to enhance feature extraction and improve the accuracy of Alzheimer's Disease detection. The system uses transfer learning, advanced architectural integrations, and optimized training techniques to achieve high performance on MRI-based datasets.

## Repository Structure
- **`DS301_Project.ipynb`**: Contains all code for data preprocessing, model implementation, training, evaluation, and visualization.
- **Dataset**: The dataset can be downloaded from [this link](https://drive.google.com/uc?id=10-b4PKd6UUTkZU3SdOn_hbgZHU8CUjT_&confirm=t&uuid=a3d8c59d-edfe-4d49-96a3-2a9246d0a4cc), which references the Kaggle source [Dataset-Alzheimer](https://www.kaggle.com/datasets/yasserhessein/dataset-alzheimer).

## Features
1. **Hybrid Architecture**:
   - Combines **ResNet50** for pretrained feature extraction and **DenseBlocks** for improved feature reuse and gradient propagation.
   - Transition blocks compress features using 1x1 convolutions and average pooling.

2. **Optimized Training Techniques**:
   - **Mixed precision training** for efficient GPU memory utilization.
   - **Cosine annealing scheduler** for learning rate adjustment.
   - **AdamW optimizer** with weight decay (0.01) for stable convergence.
   - **Dropout (0.2)** to prevent overfitting.

3. **Performance Improvements**:
   - Base ResNet50: Accuracy = 0.45
   - Transfer Learning + Health Dataset: Accuracy = 0.71
   - Hybrid Architecture (ResNet50 + DenseBlock): Accuracy = 0.75

## Usage Instructions

### Dataset
The dataset required for this project is automatically downloaded as part of the notebook execution process. Simply run the provided notebook, and the dataset will be downloaded and placed in a folder named data at the repository root.

### Running the Project
1. Open the Jupyter notebook `DS301_Project.ipynb`.
2. Install necessary dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-image scikit-learn tqdm tensorflow keras gdown torch torchvision Pillow typeguard scikeras
3. Run all cells in the notebook to preprocess the data, train the model, and visualize the results.

## Results
  - Base ResNet50 Model: Accuracy = 0.45
  - Transfer Learning with Health Dataset: Accuracy = 0.71
  - Hybrid Model (ResNet50 + DenseBlocks): Accuracy = 0.75

## Contributors
  - Yakun Du: yd2023@nyu.edu
  - Zihan Zhou: zz4029@nyu.edu

## Acknowledgments
The dataset used in this project is sourced from Kaggle and hosted on Google Drive for easier access. We appreciate the efforts of the dataset creators for providing high-quality MRI image data.
