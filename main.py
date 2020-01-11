import numpy as np
import os
from preprocessing import TitanicPreprocessing
from models import TitanicModel


def import_data(file_name: str, file_path: str = None) -> (np.array, np.array):
    if not isinstance(file_name, str):
        raise TypeError("Filename must be string")
    if file_path is None:
        file_path = "./data/"
    file_path = file_path + file_name
    return np.genfromtxt(file_path, delimiter=",", skip_header=True, filling_values=-1)


def preprocess_data(data, testset):
    return TitanicPreprocessing.preprocess_data(data, testset)


def main():
    titanic_model = TitanicModel(9, 256, 2)
    train_data = import_data("train.csv")
    test_data = import_data("test.csv")
    train_data, train_labels = preprocess_data(train_data, False)
    titanic_model.train(train_data, train_labels[:, 1], metrics=["accuracy"], epochs=20)

    test_data, test_labels = preprocess_data(test_data, True)
    predictions = titanic_model.predict(test_data)
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i, 1] == predictions[i]:
            correct += 1
    print(correct / len(test_labels))


if __name__ == '__main__':
    main()
