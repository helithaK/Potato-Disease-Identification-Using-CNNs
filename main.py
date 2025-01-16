from models.train import train_model  # Import the train_model function
from utils.visualization import plot_training  # Import the plot_training function
import argparse  # Import argparse for command-line argument parsing

# Main function to train the model and visualize results
def main():
    """
    Main function to train a CNN model using the dataset provided via command-line arguments.
    Visualizes the training and validation performance after training.
    """
    # Create an argument parser for command-line inputs
    parser = argparse.ArgumentParser()
    
    # Add argument for the path to the dataset
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    
    # Add argument for the path to save the trained model
    parser.add_argument('--save_model', type=str, required=True, help='Path to save the trained model')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Train the model using the provided dataset and save path
    history = train_model(args.data_dir, args.save_model)
    
    # Plot the training and validation performance
    plot_training(history)

# Entry point of the script
if __name__ == "__main__":
    main()
