import numpy as np
import tensorflow.keras as tf


class TitanicModel:
    def __init__(self, input_size: int, layer_size: int, output_size: int):
        self._model = self._init_model(input_size, layer_size, output_size)

    def _init_model(self, input_size: int, layer_size: int, output_size: int) -> tf.Sequential:
        model = tf.Sequential()
        model.add(tf.layers.Dense(layer_size, input_shape=(input_size,), activation="elu"))
        model.add(tf.layers.Dense(layer_size, activation="elu"))
        model.add(tf.layers.Dense(output_size, activation="elu"))
        model.add(tf.layers.Dense(output_size, activation="elu"))
        return model

    def train(self, data: np.array, labels: np.array, optimizer=tf.optimizers.Adam(learning_rate=0.01),
              loss_function=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              epochs: int = 1000, batch_size: int = 32, metrics=None):
        self._model.compile(optimizer, loss=loss_function, metrics=metrics)
        self._model.fit(data, labels, batch_size, epochs=epochs, validation_split=0.15, shuffle=True)

    def predict(self, data):
        return self._model.predict(data)
