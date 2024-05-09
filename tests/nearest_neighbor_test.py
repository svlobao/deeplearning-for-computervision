import pytest

from Image_Classification.scripts.nearest_neighbor import NearestNeighbor, load_data


def test_train():
    model = NearestNeighbor()

    train_images, train_labels, _ = load_data()

    assert model.train(images=train_images, labels=train_labels) is not None
