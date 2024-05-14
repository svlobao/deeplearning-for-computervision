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
    
------------------------------------------------------------

#TODO: 

    - Implement train + test + val splits
        - val is for tunning hyperparameters
        - test is unseen data and final metric
    - Implement evaluation metrics
        - k-fold cross-validation
    - Execute code on real dataset
    

"""

import numpy, logging, random
from typing import Union


class NearestNeighbor:
    def __init__(self) -> None:
        pass

    def _abs(self, diff: Union[int, float]) -> Union[int, float]:
        if diff < 0:
            return -diff
        else:
            return diff

    def _shape(self, arr: numpy.ndarray) -> tuple:
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input array should be of type numpy.array.\n")
        return (len(arr), len(arr[0]))

    def _count(self, combined_neighbors: list[tuple]) -> str:

        labels = list(zip(*combined_neighbors))[1]

        label_count = {}

        for label in labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        sorted_label_count = sorted(
            label_count.items(), key=lambda pair: pair[1], reverse=True
        )

        return sorted_label_count[0][0]

    def _find_neighbors(
        self, distances: list[Union[int, float]], k: int
    ) -> list[tuple]:
        if not isinstance(distances, list):
            raise TypeError(
                f"'distances' should be a list of type {list[Union[int, float]]}. Thus, the type {type(distances)} is not valid.\n"
            )
        if not len(self.x) == len(distances):
            raise IndexError(
                f"The size of training data '{len(self.x)} is different from the size of the distances '{len(distances)}'.\n"
            )

        # Will match distances with image and label, and then sort them
        # The k-th first tuples will bring the k nearest neighbors for the v-th image

        combined = list(zip(self.x, self.y, distances))
        combined = sorted(combined, key=lambda trio: trio[2])

        return combined[0:k]

    def _distance(self, method: str, arr_1: numpy.ndarray, arr_2: numpy.ndarray) -> int:
        """
        arr_1 and arr_2 should:

        - Be of type numpy.ndarray.
        - Have length >= 1

        If method == 'L1', it returns the L1 "Manhattan" distance.

        Otherwise, if method == 'L2', then it returns the L2 "Euclidean" distance.

        """

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

    def _train(self, train_data):
        self.x = list(zip(*train_data))[0]  # images
        self.y = list(zip(*train_data))[1]  # labels

        if len(self.x) != len(self.y):
            raise IndexError(
                "The number of labels is different from the number of images.\n"
            )
        if len(self.x) == 0:
            raise ValueError("Training images list is empty.\n")
        if len(self.y) == 0:
            raise ValueError("Training labels list is empty.\n")

        logging.info("5. SUCCESS: finished training the model.\n\n")

    def predict(
        self, train_data: list[tuple], val_data: list[tuple], k_neighbors: int
    ) -> dict:  # Should call predict() for every k

        if k_neighbors < 1:
            raise ValueError(
                f"It is not possible to classify for less than one neighbor. Therefore, {k_neighbors} is not a valid number for 'k'.\n"
            )
        if not isinstance(k_neighbors, int):
            raise TypeError(
                f"'k' should be an integer. Thus, the value {k_neighbors} is not valid.\n"
            )
        if len(train_data) == 0:
            raise ValueError("train_data is empty.\n")
        if len(val_data) == 0:
            raise ValueError("val_data is empty.\n")

        self._train(train_data)
        
        if k_neighbors > len(self.y):
            raise ValueError(
                f"'k' should be equal or less the total amount of training samples, which is {len(self.y)}. 'k = {k_neighbors}' was passed."
            )
        
        
        val_images = list(zip(*val_data))[0]
        val_labels = list(zip(*val_data))[1]

        predict = {}
        for v in range(len(val_images)):
            distances = []
            for t in range(len(self.x)):
                distances.append(self._distance("L1", val_images[v], self.x[t]))
            combined_neighbors = self._find_neighbors(distances, k_neighbors)

            predicted_label = self._count(combined_neighbors)
            predict[v] = (
                val_labels[v],
                predicted_label,
                predicted_label == val_labels[v],
            )

        return predict


class Data:
    def __init__(self) -> None:
        pass

    def load_raw_data(self):
        images = [
            numpy.array([[1, 1], [2, 2]]),
            numpy.array([[1, 1], [2, 1]]),
            numpy.array([[1, 3], [1, 3]]),
            numpy.array([[5, 24], [12, 11]]),
            numpy.array([[4, 2], [1, 11]]),
            numpy.array([[2, 4], [2, 1]]),
            numpy.array([[8, 10], [3, 21]]),
            numpy.array([[1, 11], [8, -11]]),
            numpy.array([[9, -14], [-10, 11]]),
            numpy.array([[9, -1], [13, 1]]),
            numpy.array([[9, -3], [1, 5]]),
            numpy.array([[3, -1], [0, 1]]),
        ]

        labels = [
            "1",
            "2",
            "3",
            "1",
            "2",
            "3",
            "1",
            "2",
            "3",
            "1",
            "2",
            "3",
        ]

        return list(zip(images, labels))

    def split(self, input: list[tuple[numpy.ndarray, str]], train: int, val: int, test: int):
        """
        - 'input': is a list whose elements is a tuple of an image and its label
        - 'train', 'val' and 'test' are the portions for each batch size in percentage of the original data. Example: 80, 10, 10.
        """

        if len(input) == 0:
            raise ValueError("Input list of images and labels is empty.\n")
        if train + val + test != 100:
            raise ValueError(
                f"I thought it was needless to say, but the percentages of train, val and test need to sum up to 100. You provided {train+val+train}. F.\n"
            )

        _total_sample_size = len(input)
        _train_sample_size = int(_total_sample_size * (train / 100))
        _val_sample_size = int(_total_sample_size * (val / 100))
        _test_sample_size = int(_total_sample_size * (test / 100))

        # Derive exact values
        count = [0, 0, 0]
        while (
            _train_sample_size + _val_sample_size + _test_sample_size
            < _total_sample_size
        ):
            if count == [0, 0, 0]:
                _test_sample_size += 1
                count[2] += 1
                continue

            if count == [0, 0, 1]:
                _val_sample_size += 1
                count[1] += 1
                continue

            if count == [0, 1, 1]:
                _train_sample_size += 1
                count = [0, 0, 0]
                continue

        while (
            _train_sample_size + _val_sample_size + _test_sample_size
            > _total_sample_size
        ):
            if count == [0, 0, 0]:
                _train_sample_size -= 1
                count[0] += 1
                continue

            if count == [1, 0, 0]:
                _val_sample_size -= 1
                count[1] += 1
                continue

            if count == [1, 1, 0]:
                _test_sample_size -= 1
                count = [0, 0, 0]
                continue

        assert (
            _train_sample_size + _val_sample_size + _test_sample_size
            == _total_sample_size
        )

        random.shuffle(input)
        _train_samples = input[:_train_sample_size]
        _val_samples = input[_train_sample_size : _train_sample_size + _val_sample_size]
        _test_samples = input[
            _train_sample_size
            + _val_sample_size : _train_sample_size
            + _val_sample_size
            + _test_sample_size
        ]

        return [_train_samples, _val_samples, _test_samples]


def log_init():
    logging.basicConfig(
        filename="/Users/sandrolobao/Documents/Computer Vision/Deep Learning for CV/Image_Classification/log/nearest_neighbor.log",
        level=logging.DEBUG,
    )
    logging.info("\n\nInitiated logging...\n\n")


if __name__ == "__main__":

    log_init()

    nn = NearestNeighbor()

    data = Data()
    input = data.load_raw_data()
    logging.info("1. Loaded test images.\n\n")

    train_data, val_data, test_data = data.split(input, 70, 15, 15)

    predict = []
    k_neighbors = [3]

    # validate model on val_data
    for k in k_neighbors:
        predict.append(nn.predict(train_data, val_data, k))

    print(predict)
