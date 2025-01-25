import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Model
from keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
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

def train_resnet():
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(15, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.layers[-3].name = 'features'

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(
    k=3, name="top_k_categorical_accuracy", dtype=None
)])

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=30,
        callbacks=[
            ModelCheckpoint('KUBA3.keras', save_best_only=True),
            EarlyStopping(patience=5, restore_best_weights=True)
        ])
    
    return history

if __name__ == '__main__':
    history = train_resnet()
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('resnet-history.csv', index=False)