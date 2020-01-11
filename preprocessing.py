import numpy as np
from enum import Enum


class TitanicPreprocessing:
    @staticmethod
    def preprocess_data(data: np.array, testset, remove_ids: bool = True, remove_names: bool = True):
        if testset:
            print(data.shape)
            np.hstack((np.zeros((len(data), 1)), data))
            print(data.shape)
        processed_data = TitanicPreprocessing.convert_categories_to_ints(data)
        processed_data = TitanicPreprocessing.normalize_data(processed_data)
        processed_data, labels = TitanicPreprocessing.strip_columns(processed_data)

        return processed_data, labels

    @staticmethod
    def convert_categories_to_ints(data):
        # data[TitanicColumns.Sex.value] = np.where(data[:, TitanicColumns.Sex.value] == "male",
        #                                           Sex.Male.value, Sex.Female.value)
        # TODO: Get rid of this for loop
        for i in range(len(data)):
            value = data[i, TitanicColumns.Sex.value]
            if value == "male":
                data[i, TitanicColumns.Sex.value] = 1
            elif value == "female":
                data[i, TitanicColumns.Sex.value] = 2
            else:
                data[i, TitanicColumns.Sex.value] = 3

            value = data[i, TitanicColumns.Embarked.value]
            if value == "Q":
                data[i, TitanicColumns.Embarked.value] = 1
            elif value == "S":
                data[i, TitanicColumns.Embarked.value] = 2
            elif value == "C":
                data[i, TitanicColumns.Embarked.value] = 3
            else:
                data[i, TitanicColumns.Embarked.value] = 4

            value = data[i, TitanicColumns.Cabin.value]
            if value == -1:
                data[i, TitanicColumns.Cabin.value] = 1
            elif value == "G6":
                data[i, TitanicColumns.Cabin.value] = 2
            else:
                data[i, TitanicColumns.Cabin.value] = 3
        return data

    @staticmethod
    def strip_columns(data, column_number=0):
        labels = data[:, TitanicColumns.PassengerId.value:TitanicColumns.Survived.value + 1]
        data = np.delete(data, TitanicColumns.Name.value, 1)
        data = np.delete(data, TitanicColumns.Ticket.value - 1, 1)
        return data[:, 2:], labels

    @staticmethod
    def normalize_data(data):
        data[TitanicColumns.Pclass.value] /= 3.0
        data[TitanicColumns.Sex.value] /= 3.0
        data[TitanicColumns.Age.value] /= 80.0
        data[TitanicColumns.SibSp.value] /= 8.0
        data[TitanicColumns.Parch.value] /= 6.0
        data[TitanicColumns.Fare.value] /= 512.0
        data[TitanicColumns.Embarked.value] /= 4.0
        return data


class TitanicColumns(Enum):
    PassengerId = 0
    Survived = 1
    Pclass = 2
    Name = 3
    Sex = 4
    Age = 5
    SibSp = 6
    Parch = 7
    Ticket = 8
    Fare = 9
    Cabin = 10
    Embarked = 11


class Sex(Enum):
    Male = 1
    Female = 2
