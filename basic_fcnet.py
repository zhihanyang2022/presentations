import sys
sys.stderr = open('/dev/null', 'w')

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

### DATA PREPROCESSING ###

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

### MODEL CONSTRUCTION ###

model = models.Sequential()
model.add(layers.Dense(118, activation='relu', input_shape=(28*28,), trainable=True))
model.add(layers.Dense(10, activation='softmax', trainable=True))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

### MODEL TRAINING ###

model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=1)

### MODEL EVALUATION ###

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

### SUMMARY ###

print(model.summary())
print("Test accuracy: {0} % ".format(round(test_acc*100, 2)))

# print()
#
# print("Weights")
# first_layer_weights = model.layers[0].get_weights()
# print(first_layer_weights[0].shape)
# print(first_layer_weights[1].shape)
# print(first_layer_weights[2].shape)
# print()
#
# print("Biases")
# first_layer_biases  = model.layers[0].get_weights()[1]
# print(first_layer_biases)
