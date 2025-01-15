import os
from models.model import create_model
from data.preprocess_data import load_data

def train_model(data_dir, save_model_path):
    train_gen, val_gen = load_data(data_dir)
    input_shape = train_gen.image_shape
    num_classes = len(train_gen.class_indices)

    model = create_model(input_shape, num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save(save_model_path)
    return history