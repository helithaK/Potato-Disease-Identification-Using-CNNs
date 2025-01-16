from models.train import train_model
from utils.visualization import plot_training
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_model', type=str, required=True, help='Path to save the trained model')
    args = parser.parse_args()

    history = train_model(args.data_dir, args.save_model)
    plot_training(history)

if __name__ == "__main__":
    main()
