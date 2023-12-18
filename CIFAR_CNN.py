import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.datasets import cifar10

# Define the hyperparameters
BATCH_SIZE = 32
EPOCHS = 100

DATA_AUGMENTATION = True

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = x_train/255
x_test = x_test/255

# Define the model


def create_model(data_augmentation):
    model = Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Resizing(75, 75))
    if data_augmentation:
        model.add(tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal", input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.1))

    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model


def run_experiment():
    model = create_model(data_augmentation=DATA_AUGMENTATION)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # Train the model.
    _ = model.fit(x=x_train, y=y_train, epochs=EPOCHS,
                  validation_data=(x_test, y_test), batch_size=BATCH_SIZE)

    _, accuracy = model.evaluate(x=x_test, y=y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    model.save("cifar_cnn.keras")


model = run_experiment()
