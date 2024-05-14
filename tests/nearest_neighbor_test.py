import pytest

from Image_Classification.scripts.nearest_neighbor import NearestNeighbor, Data


def test_train():
    model = NearestNeighbor()
    data = Data()
    input = data.load_raw_data()

    train_size, val_size, test_size = [70, 15, 15]
    train_data, val_data, test_data = data.split(
        input,
        train_size,
        val_size,
        test_size,
    )

    model._train(train_data)

    assert len(model.x) == len(
        model.y
    ), "The length of images and labels should be equal.\n"

    assert len(model.x) != 0, "The length of images and labels should not be zero.\n"
