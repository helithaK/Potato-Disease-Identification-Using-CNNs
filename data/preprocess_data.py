import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load image data from a directory
def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load and preprocess image data for training and validation using Keras ImageDataGenerator.
    
    Args:
        data_dir (str): Path to the directory containing image datasets organized in subdirectories by class.
        img_size (tuple): Target size to resize images (width, height). Default is (224, 224).
        batch_size (int): Number of images to include in each batch. Default is 32.
        
    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
    """
    # Create an ImageDataGenerator object for preprocessing the images
    # - Rescales pixel values to the range [0, 1] by dividing by 255
    # - Splits data into training (80%) and validation (20%) subsets
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    
    # Create a generator for training data
    train_gen = datagen.flow_from_directory(
        data_dir,                 # Path to the main dataset directory
        target_size=img_size,     # Resize all images to the specified size
        batch_size=batch_size,    # Number of images per batch
        class_mode='categorical', # Labels are categorical (one-hot encoded)
        subset='training'         # Use the training subset of the data
    )
    
    # Create a generator for validation data
    val_gen = datagen.flow_from_directory(
        data_dir,                 # Path to the main dataset directory
        target_size=img_size,     # Resize all images to the specified size
        batch_size=batch_size,    # Number of images per batch
        class_mode='categorical', # Labels are categorical (one-hot encoded)
        subset='validation'       # Use the validation subset of the data
    )
    
    # Return both the training and validation data generators
    return train_gen, val_gen
