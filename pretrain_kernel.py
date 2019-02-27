# <codecell>
import sys
stdout = sys.stderr
sys.stderr = open('/dev/null', 'w')

import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

%matplotlib inline

# <codecell>

def pretrain_kernels(train_images, start, end, train_labels, pretrained_kernels=None):

    n_rows = 28
    n_cols= 28
    threshold = 240

    if pretrained_kernels==None:
        pretrained_kernels = [np.zeros((n_rows, n_cols)) for i in range(10)]

    for image, label in zip(train_images[start:end], train_labels[start:end]):
        for i in range(n_rows):
            for j in range(n_cols):
                if (image[i][j] > threshold):
                    pretrained_kernels[label][i][j] += 1

    return pretrained_kernels

# <codecell>

pretrained_kernels = pretrain_kernels(train_images, 0, 0, train_labels)

# <codecell>

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5, 5))
pretrained_kernels = pretrain_kernels(train_images, 0, 100, train_labels)
ax1.matshow(pretrained_kernels[1])
pretrained_kernels = pretrain_kernels(train_images, 0, 500, train_labels)
ax2.matshow(pretrained_kernels[1])
pretrained_kernels = pretrain_kernels(train_images, 0, 1000, train_labels)
ax3.matshow(pretrained_kernels[1])

ax1.savefig("pretrained_kernel_1_100.png")
ax2.savefig("pretrained_kernel_1_500.png")
ax3.savefig("pretrained_kernel_1_10000.png")

# <codecell>
results = []
wrong_classifications = []
for image, label in zip(test_images[:10000], test_labels[:10000]):
    activations = [np.sum(np.multiply(image, pretrained_kernels[i])) for i in range(10)]
    classification = np.argmax(activations)
    if classification == label:
        results.append(1)
    else:
        results.append(0)
        wrong_classifications.append(image)

print("Test accuracy: {0}".format(np.mean(results)))

# <codecell>
for index, image in enumerate(wrong_classifications):
    plt.matshow(image)
    plt.savefig("pretrained_fcn_mistakes {0}".format(index))
