import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Rock_and_Mine_BinaryClassification:
    # loading dataset to pandas data frame
    sonar_data = pd.read_csv('data/sonar_data.csv', header=None)
    X=[]
    y=[]

    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]

    # creating a logistic regression model
    model = LogisticRegression()

    def data_collection(self):
        # getting first five records
        print(self.sonar_data.head())
        print()

        # dimensions
        print(self.sonar_data.shape)
        print()

        # statistical analysis
        print(self.sonar_data.describe())
        print()

        # checking missing values
        print(self.sonar_data.isnull().sum())
        print()

        # number of data cases for rock(r) and mines (m)
        print(self.sonar_data[60].value_counts())
        print()

        # mean values for all the attributes over the mine and rock data cases
        print(self.sonar_data.groupby(60).mean())
        print()
    
    def data_preprocessing(self):
        # separating data and label
        self.X = self.sonar_data.drop(60, axis=1) # axis removes the rows
        self.y = self.sonar_data[60]

        print(self.X)
        print()
        print(self.y)
        print()

        # splitting data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=2)
        # the stratify parameter is used to consider equal amount of data cases from each label during the split

        print(self.X_test.shape, self.X_train.shape)
        print()

    def model_training(self):        
        # training the model with training data
        self.model.fit(self.X_train, self.y_train)
    
    def model_evaluation(self):
        # accuracy on the training data
        X_train_pred = self.model.predict(self.X_train)
        training_data_accuracy = accuracy_score(X_train_pred, self.y_train)
        print("Training Accuracy: ", training_data_accuracy)
        print()

        # accuracy on the test data
        X_test_pred = self.model.predict(self.X_test)
        testing_data_accuracy = accuracy_score(X_test_pred, self.y_test)
        print("Testing Accuracy: ", testing_data_accuracy)
        print()
    
    def prediction(self, input_data):
        prediction = self.model.predict(input_data)
        print("Predicted Output: ", prediction)
        print()

        if prediction[0] == 'R':
            print("The object is a rock.")
        else:
            print("The object is a mine.")


if __name__ == '__main__':
    classifier = Rock_and_Mine_BinaryClassification()
    classifier.data_collection()
    classifier.data_preprocessing()
    classifier.model_training()
    classifier.model_evaluation()

    # Making a prediction
    input_data = np.asarray((
        0.0240,0.0218,0.0324,0.0569,0.0330,0.0513,0.0897,0.0713,0.0569,0.0389,0.1934,0.2434,0.2906,0.2606,0.3811,0.4997,0.3015,0.3655,0.6791,0.7307,0.5053,0.4441,0.6987,0.8133,0.7781,0.8943,0.8929,0.8913,0.8610,0.8063,0.5540,0.2446,0.3459,0.1615,0.2467,0.5564,0.4681,0.0979,0.1582,0.0751,0.3321,0.3745,0.2666,0.1078,0.1418,0.1687,0.0738,0.0634,0.0144,0.0226,0.0061,0.0162,0.0146,0.0093,0.0112,0.0094,0.0054,0.0019,0.0066,0.0023
    ))

    # reshaping the numpy array as we are predicting for one instance: passing 1, -1 indicates that there is one instance and we are predicting the label for one instance
    input_data = input_data.reshape(1, -1)

    classifier.prediction(input_data)




