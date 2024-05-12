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

import numpy, logging
from typing import Union


class NearestNeighbor:
    def __init__(self) -> None:
        pass

    def _abs(self, diff: Union[int, float]) -> Union[int, float]:
        logging.info("8. Called _abs().\n\n")
        if diff < 0:
            return -diff
        else:
            return diff

    def _shape(self, arr: numpy.ndarray) -> tuple:
        logging.info("7. Called _shape().\n\n")
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input array should be of type numpy.array.\n")
        return (len(arr), len(arr[0]))

    def _find_neighbors(
        self, distances: list[Union[int, float]], k: int
    ) -> list[tuple]:
        logging.info("8. Called _find_neighbors().\n\n")
        if not isinstance(distances, list):
            raise TypeError(
                f"'distances' should be a list of type {list[Union[int, float]]}. Thus, the type {type(distances)} is not valid.\n"
            )
        if not len(self.x) == len(distances):
            raise IndexError(
                f"The size of training data '{len(self.x)} is different from the size of the distances '{len(distances)}'.\n"
            )

        combined = list(zip(self.x, self.y, distances))
        combined.sort(key=lambda x: x[2])
        return combined[0:k]

    def _distance(self, arr_1: numpy.ndarray, arr_2: numpy.ndarray, method: str) -> int:
        """
        arr_1 and arr_2 should:

        - Be of type numpy.ndarray.
        - Have length >= 1

        If method == 'L1', it returns the L1 "Manhattan" distance.

        Otherwise, if method == 'L2', then it returns the L2 "Euclidean" distance.

        """

        logging.info("6. Called _distance().\n\n")

        methods = ["L1", "L2"]
        distance = 0

        if self._shape(arr_1) != self._shape(arr_2):
            raise IndexError("The dimensions of both matrices are not equal.\n")
        if len(arr_1) == 0:
            raise IndexError("Dimension of matrices cannot be 0.\n")
        if method not in methods:
            raise ValueError(f"Method not recognized. Try one of: {methods}.\n")

        if method == "L1":
            for i in range(len(arr_1)):
                for j in range(len(arr_1[0])):
                    distance += self._abs(arr_1[i][j] - arr_2[i][j])
            return distance

        if method == "L2":
            for i in range(len(arr_1)):
                for j in range(len(arr_1[0])):
                    distance += (arr_1[i][j] - arr_2[i][j]) ** 2
            return distance**0.5

    def _train(self):
        logging.info("4. Called _train().\n\n")
        self.x = load_data()[0]  # images
        self.y = load_data()[1]  # labels

        if len(self.x) != len(self.y):
            raise IndexError(
                "The number of labels is different from the number of images.\n"
            )
        if len(self.x) == 0:
            raise ValueError("Training images list is empty.\n")
        if len(self.y) == 0:
            raise ValueError("Training labels list is empty.\n")

        logging.info("5. SUCCESS: finished training the model.\n\n")

    def predict(self, test_image: numpy.ndarray, k: int) -> dict:
        logging.info("3. Called predict().\n\n")
        if k < 1:
            raise ValueError(
                f"It is not possible to classify for less than one neighbor. Therefore, {k} is not a valid number for 'k'.\n"
            )
        if not isinstance(k, int):
            raise TypeError(
                f"'k' should be an integer. Thus, the value {k} is not valid.\n"
            )

        self._train()

        distances = []
        for img in self.x:
            distances.append(self._distance(img, test_image, "L1"))
        logging.info("7. SUCCESS: computed all distances.\n\n")

        combined = self._find_neighbors(distances, k)
        logging.info(f"8. SUCCESS: found all {k} neighbors: \n\n{combined}\n\n")

        pred = {}
        _, labels, _ = zip(*combined)
        logging.info(f"9. Decompacting labels: \n{labels}\n\n")

        print("\n\nLabels: ", labels, "\n\n")
        for label in labels:
            if label not in pred:
                pred[label] = 1
            else:
                pred[label] += 1

        sorted_pred = dict(sorted(pred.items(), key=lambda pair: pair[1], reverse=True))

        logging.info(
            f"10. SUCCESS: finished predicting. The given test image is labeled as: {list(sorted_pred.keys())[0]}\n\nAlso, here's the distance vector: \n\n{distances}\n\n"
        )

        return list(sorted_pred.keys())[0]


def load_data():
    train_images = [
        numpy.array([[1, 1], [2, 2]]),
        numpy.array([[1, 3], [1, 3]]),
        numpy.array([[5, 24], [12, 11]]),
        numpy.array([[4, 2], [1, 11]]),
        numpy.array([[2, 4], [2, 1]]),
        numpy.array([[8, 10], [3, 21]]),
        numpy.array([[1, 11], [8, -11]]),
        numpy.array([[9, -14], [-10, 11]]),
        numpy.array([[9, -1], [13, 1]]),
        numpy.array([[2, 54], [12, 11]]),
    ]

    train_labels = [
        "image_1",
        "image_2",
        "image_3",
        "image_1",
        "image_2",
        "image_3",
        "image_1",
        "image_2",
        "image_3",
        "image_3",
    ]

    test_images = [
        numpy.array([[5, 24], [12, 11]]),
        # numpy.array([[1, 5], [1, 5]]),
        # numpy.array([[0, 0], [0, 0]]),
        # numpy.array([[2, 54], [12, 11]]),
        # numpy.array([[1, 11], [8, -11]]),
    ]

    return (
        train_images,
        train_labels,
        test_images,
    )


def log_init():
    logging.basicConfig(
        filename="/Users/sandrolobao/Documents/Computer Vision/Deep Learning for CV/Image_Classification/log/nearest_neighbor.log",
        level=logging.DEBUG,
    )
    logging.info("\n\nInitiated logging...\n\n")


if __name__ == "__main__":

    log_init()

    _, _, test_images = load_data()
    logging.info("1. Loaded test images.\n\n")

    nn = NearestNeighbor()

    predict = []
    # k_neighbors = [1, 2, 3, 4, 5]
    k_neighbors = [3]

    for k in k_neighbors:
        for img in test_images:
            logging.info(f"2. Currently testing image: \n\n{img}\n\n")
            predict.append(
                nn.predict(img, k),
            )

    # print(predict)
