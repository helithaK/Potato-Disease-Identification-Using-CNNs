from sklearn.metrics import classification_report, confusion_matrix
import numpy as np  # Ensure numpy is imported for processing predictions

# Function to evaluate a trained model on a given dataset
def evaluate_model(model, data_gen):
    """
    Evaluate a trained model using a data generator.
    Prints the classification report and confusion matrix.
    
    Args:
        model: Trained Keras model to be evaluated.
        data_gen: Data generator for evaluation, typically validation or test data.
        
    Returns:
        None
    """
    # True class labels from the data generator
    y_true = data_gen.classes  # Array of true labels corresponding to the dataset
    
    # Model predictions on the data generator
    y_pred = model.predict(data_gen)  # Output probabilities for each class
    
    # Convert predicted probabilities to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)  # Choose the class with the highest probability for each sample

    # Print the classification report
    # - Includes precision, recall, F1-score, and support for each class
    # - target_names maps class indices to their respective class names
    print(classification_report(y_true, y_pred_classes, target_names=data_gen.class_indices.keys()))
    
    # Print the confusion matrix
    # - Compares true labels to predicted labels
    print(confusion_matrix(y_true, y_pred_classes))
