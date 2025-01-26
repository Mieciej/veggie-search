import pandas as pd
import tensorflow as tf
import keras
import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import Model
from keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(0.2),
])

train_dir = 'veggie-images/train'
validation_dir = 'veggie-images/validation'

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

def create_custom_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    
    x = Flatten()(x)
    features = Dense(1024, activation='relu', name='features')(x)
    x = Dropout(0.5)(features)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(15, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_custom_model():
    model = create_custom_model(input_shape=(224, 224, 3))
    model.layers[-5].name = 'features'
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(
        k=3, name="top_k_categorical_accuracy", dtype=None
    )
    ])

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=45,
        callbacks=[
            ModelCheckpoint('vgg16-scratch-model-best.keras', save_best_only=True),
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    return history

if __name__ == "__main__":
    history = train_custom_model()
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('custom-model-history.csv', index=False)