import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

image_data = np.random.rand(1, 4, 4, 1).astype('float32')
model = models.Sequential([
    layers.Conv2D(
        filters=1, kernel_size=(2, 2), strides=(1, 1),
        use_bias=False, padding='valid', input_shape=(4, 4, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Flatten()
])
model.summary()
_ = model(image_data)
inputs = layers.Input(shape=(4, 4, 1))
x = layers.Conv2D(
    filters=1, kernel_size=(2, 2), strides=(1, 1), use_bias=False, padding='valid')(inputs)
conv_model = models.Model(inputs=inputs, outputs=x)
conv_output = conv_model.predict(image_data)
print("Conv output shape:", conv_output.shape)
maxpool_model = models.Model(inputs=inputs, outputs=layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x))
maxpool_output = maxpool_model.predict(image_data)
print("Maxpool output shape:", maxpool_output.shape)
flatten_model = models.Model(inputs=inputs, outputs=layers.Flatten()(x))
flatten_output = flatten_model.predict(image_data)
print("Flatten output shape:", flatten_output.shape)

