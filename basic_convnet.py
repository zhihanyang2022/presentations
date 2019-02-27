from keras import layers
from keras import models

model = models.Sequential()

# cnn
# - input_shape=(image_height, image_width, image_channels)
# - a convolution layer is defined by 2 key parameters
#   - the size of kernels / the size of patches, e.g. (3, 3) or (5, 5)
#   - the number of kernels / depth output feature maps
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# output shape: (26, 26, 32), every dimension in the depth axis represents a kernel
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# classifier on top of cnn
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# import data set
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) # normalize
train_images = train_images.astype('float32') / 255 # normalize

test_images = test_images.reshape((10000, 28, 28, 1)) # check shape
test_images = test_images.astype('float32') / 255 # normalize

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(train_images, train_labels, epochs=5, batch_size=64)

#test_loss, test_acc = model.evaluate(test_images, test_labels)
print(model.summary())
#print(test_acc) # about 99%, almost to state-of-the-art
