import matplotlib.pyplot as plt

# Function to plot training and validation accuracy and loss
def plot_training(history):
    """
    Plot training and validation accuracy and loss over epochs.

    Args:
        history: History object returned by the model.fit() method.
        
    Returns:
        None
    """
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)  # Create a subplot (1 row, 2 columns, first plot)
    plt.plot(history.history['accuracy'], label='Train Accuracy')  # Training accuracy
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Validation accuracy
    plt.legend()  # Display legend
    plt.title('Accuracy')  # Title of the plot
    
    # Plot loss
    plt.subplot(1, 2, 2)  # Create a subplot (1 row, 2 columns, second plot)
    plt.plot(history.history['loss'], label='Train Loss')  # Training loss
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validation loss
    plt.legend()  # Display legend
    plt.title('Loss')  # Title of the plot
    
    # Show the plots
    plt.show()
