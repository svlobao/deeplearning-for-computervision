"""
Implement Nearest Neighbor algorithm from scratch

Sandro V. Lobão

May 6th, 2024

------------------------------------------------------------

Algorithm:

    - Store full training data (images and labels)
    - Take one test image at a time
    - Perform the L1 distance with each image of the training data
    - Return the index of the image with minimum L1 distance

"""

import numpy


class NearestNeighbor:
    def __init__(self) -> None:
        pass

    def _absolute_element_sum(self, diff: numpy.ndarray) -> int:
        result = 0

        # print("\n", diff, "\n")

        for row in diff:
            for element in row:
                if element < 0:
                    result += -element
                else:
                    result += element

        # print("\n", result, "\n")

        return result

    def _shape(self, arr: numpy.ndarray) -> tuple:
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input array should be of type numpy.array.\n")

        # print("\n", (len(arr), len(arr[0])), "\n", numpy.shape(arr))

        return (len(arr), len(arr[0]))

    def _L1_distance(self, arr_1: numpy.ndarray, arr_2: numpy.ndarray) -> int:
        """
        arr_1 and arr_2 should:

        - Be of type numpy.ndarray.
        - Have length >= 1

        returns the scalar (int) that represents the L1 "Manhattan" distance.

        """
        if self._shape(arr_1) != self._shape(arr_2):
            raise IndexError("The dimensions of both matrices are not equal.\n")
        if len(arr_1) == 0:
            raise IndexError("Dimension of matrices cannot be 0.\n")
        distance = self._absolute_element_sum(arr_1 - arr_2)

        # print(distance)

        return distance

    def _min(self, arr: numpy.ndarray) -> int:
        """
        This method returns the index of the minimum value in an array.

        If two or more minimum distances are found, then it will return the last seen index
        """
        if len(arr) == 0:
            raise ValueError("Input array is empty.\n")
        min_value = arr[0]
        curr = 0
        for i in range(len(arr)):
            if arr[i] < min_value:
                curr = i
        # print(curr)
        return curr

    def train(self, images: list, labels: list) -> None:
        self.x = images
        self.y = labels
        # print("\n", self.x, "\n\n", self.y, "\n\n")

        if len(self.x) != len(self.y):
            raise IndexError(
                "The number of labels is different from the number of images.\n"
            )

    def predict(self, test_image: numpy.ndarray) -> tuple[int, str]:
        distances = []
        for img in self.x:
            distances.append(self._L1_distance(img, test_image))
        return (self._min(distances), self.y[self._min(distances)])


if __name__ == "__main__":
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
    nn.train(images=train_images, labels=train_labels)
    predict = []
    for img in test_images:
        predict.append(
            nn.predict(test_image=img),
        )

    print(predict)
