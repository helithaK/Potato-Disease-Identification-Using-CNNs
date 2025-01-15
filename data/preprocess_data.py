import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        data_dir, 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir, 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        subset='validation'
    )
    return train_gen, val_gen