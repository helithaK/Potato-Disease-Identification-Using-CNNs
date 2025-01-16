import os
from models.model import create_model  # Import the create_model function from the specified module
from data.preprocess_data import load_data  # Import the load_data function from the specified module

# Function to train a model
def train_model(data_dir, save_model_path):
    """
    Train a CNN model on image data and save the trained model.
    
    Args:
        data_dir (str): Path to the directory containing image datasets organized in subdirectories by class.
        save_model_path (str): Path to save the trained model.
        
    Returns:
        history: Training history object containing loss and accuracy metrics for each epoch.
    """
    # Load training and validation data generators
    train_gen, val_gen = load_data(data_dir)
    
    # Get the input shape of the images from the training generator
    # Example: (224, 224, 3) for RGB images of size 224x224
    input_shape = train_gen.image_shape
    
    # Get the number of classes from the training generator
    # Example: If there are 3 subdirectories (classes), num_classes will be 3
    num_classes = len(train_gen.class_indices)

    # Create the model using the create_model function
    model = create_model(input_shape, num_classes)

    # Train the model
    # - Uses the training generator for input data
    # - Validation data is used to monitor the model's performance on unseen data
    # - Training for 10 epochs
    history = model.fit(
        train_gen,            # Training data generator
        validation_data=val_gen,  # Validation data generator
        epochs=10             # Number of training epochs
    )

    # Save the trained model to the specified path
    model.save(save_model_path)

    # Return the training history object
    return history
