from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to create a Convolutional Neural Network (CNN) model
def create_model(input_shape, num_classes):
    """
    Create a CNN model using Keras Sequential API.
    
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes for classification.
        
    Returns:
        model: Compiled Keras model.
    """
    # Initialize the Sequential model
    model = Sequential([
        # First convolutional layer
        # - 32 filters of size 3x3
        # - 'relu' activation function introduces non-linearity
        # - input_shape specifies the shape of the input images
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        
        # First max-pooling layer
        # - Reduces spatial dimensions by taking the maximum value in 2x2 windows
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        # - 64 filters of size 3x3
        # - 'relu' activation function
        Conv2D(64, (3, 3), activation='relu'),
        
        # Second max-pooling layer
        # - Again reduces spatial dimensions using 2x2 windows
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer
        # - Converts the 2D feature maps into a 1D vector for the fully connected layers
        Flatten(),
        
        # Fully connected (dense) layer
        # - 128 units with 'relu' activation
        Dense(128, activation='relu'),
        
        # Dropout layer
        # - Regularization technique to reduce overfitting
        # - Randomly sets 50% of input units to 0 during training
        Dropout(0.5),
        
        # Output layer
        # - Number of units equal to num_classes
        # - 'softmax' activation function outputs probabilities for each class
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    # - Optimizer: 'adam' (adaptive optimizer for efficient training)
    # - Loss: 'categorical_crossentropy' (used for multi-class classification)
    # - Metrics: 'accuracy' (to monitor classification performance)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Return the compiled model
    return model
