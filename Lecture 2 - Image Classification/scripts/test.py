import numpy

from nearest_neighbor import NearestNeighbor

train_images = [
    numpy.array([[1, 1], [2, 2]]),
    numpy.array([[1, 3], [1, 3]]),
    numpy.array([[5, 24], [12, 11]]),
]

train_labels = [
    "image_1",
    "image_2",
    "image_3",
]

test_images = [
    numpy.array([[1, 1], [2, 2]]),
    numpy.array([[1, 5], [1, 5]]),
    numpy.array([[0, 0], [0, 0]]),
]


nn = NearestNeighbor()

# nn.train(images=train_images, labels=train_labels)

# nn._absolute_element_sum(test_images[2] - train_images[0])

# nn._shape(train_images[1])

# nn._L1_distance(test_images[2], train_images[0])

# nn._min(numpy.array([1,4,3,5,5,6,0,1,1,0,3]))