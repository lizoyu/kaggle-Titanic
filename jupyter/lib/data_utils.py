import csv
import numpy as np
from random import shuffle

def load_Titanic(filename='../data/train - processed.csv', test=False):
    """
    load all the data from Titanic dataset

    Input:
        - filename: path to the csv file
    Output:
        - data: list of data
    """
    with open(filename, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        # split labels and data and save in dictionary
        data = []
        labels = []
        for row in fileToRead:
            if not test:
                labels.append(row.pop(0))
            data.append(row)

    if not test:
        return np.array(data), np.array(labels)
    else:
        return np.array(data)

def create_submission(model, X_test, save_path):
    """
    use trained model to create submission for Kaggle competition

    Inputs:
        - model: trained model
        - test_path: the path to the test data
        - save_path: the path to save the submission with file name
        - fit: transform the test data to fit the model
    """

    # predict and save
    predictions = model.predict(X_test)
    with open(save_path, 'wt') as csvfile:
        fileToWrite = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        # write the header
        fileToWrite.writerow(['PassengerId', 'Survived'])
        # write the predictions
        for i in range(len(predictions)):
            fileToWrite.writerow([i+892, predictions[i]])

def tester():
	x, y = load_Titanic()
	print(x.shape,y.shape)
