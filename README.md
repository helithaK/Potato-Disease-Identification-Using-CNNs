# Potato Disease Classification using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify potato diseases using image data. The dataset used for this project is sourced from Kaggle and contains labeled images of potato leaves representing different disease categories. The primary objective is to develop a model that can accurately classify the health condition of potato plants to aid in agricultural disease management.

## Dataset
The dataset used for this project can be found on Kaggle:
[Potato Dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset)

### Dataset Details:
- **Categories**: Healthy, Early Blight, Late Blight
- **Format**: Images (JPG/PNG)
- **License**: CC0-1.0

## Installation and Setup
Follow the steps below to set up and run the project.

### Prerequisites
- Python 3.7 or higher
- Kaggle API key

### Steps to Setup:
1. Install required Python libraries:
   ```bash
   pip install tensorflow keras numpy matplotlib kaggle
   ```

2. Download the dataset from Kaggle:
   - Place the `kaggle.json` API key file in the appropriate directory.
   - Run the following commands to download and extract the dataset:
     ```python
     !mkdir ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     !kaggle datasets download -d faysalmiah1721758/potato-dataset
     ```
     After downloading, extract the dataset:
     ```python
     import zipfile
     zip_path = "potato-dataset.zip"
     extract_path = "./potato-dataset"
     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
         zip_ref.extractall(extract_path)
     print("Dataset extracted!")
     ```

3. Prepare the environment by organizing the dataset for training and testing.

## Model Implementation
This project uses a CNN architecture designed to classify potato leaf images. The following steps are implemented:

1. **Data Preprocessing:**
   - Resizing images to a fixed size.
   - Normalizing pixel values.
   - Splitting the dataset into training, validation, and testing sets.

2. **Model Architecture:**
   - Convolutional layers with ReLU activation.
   - Pooling layers to reduce spatial dimensions.
   - Fully connected layers for classification.
   - Dropout layers to prevent overfitting.

3. **Training:**
   - Compile the model using Adam optimizer and categorical crossentropy loss.
   - Train with a defined number of epochs and batch size.

4. **Evaluation:**
   - Evaluate the model on the test set.
   - Generate accuracy and loss metrics.



