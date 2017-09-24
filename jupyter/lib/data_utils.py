import csv
import numpy as np
from random import shuffle

def load_Titanic(filename='../data/all.csv'):
    """
    load all the data from Titanic dataset

    Input:
        - filename: path to the csv file
    Output:
        - x_train, y_train, x_test
    """
    with open(filename, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        x_train = []; x_test = []
        y_train = []
        for row in fileToRead:
            label = row.pop(0)
            if label == 'NA':
                x_test.append(row)
            else:
                y_train.append(label)
                x_train.append(row)

    return np.array(x_train), np.array(y_train), np.array(x_test)

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
	x_train, y_train, x_test = load_Titanic('../../data/all.csv')
	print(x_train.shape,y_train.shape,x_test.shape)
