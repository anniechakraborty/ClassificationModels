import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

class Diabetes_Prediction:
    # Data Source: Kaggle (https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction)
    data = pd.read_csv('data/diabetes.csv')
    X = []
    y = []

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Creating a Support Vector Machine Classifier model
    model = svm.SVC(kernel='linear')

    def data_analysis(self):
        print('Top 5 rows :')
        print(self.data.head())
        print()
        print('Data shape :')
        print(self.data.shape)
        print()
        print('Columns, non-null entries and data types :')
        print(self.data.info())
        print()
        print('Statistical summary of the data :')
        print(self.data.describe())
        print()
        print('Count of null values in each column :')
        print(self.data.isnull().sum())
        print()
        print('Correlation matrix :')
        print(self.data.corr())
        print()
        print('Number of patients with diabetes :', self.data['Outcome'].value_counts()[1])
        print('Number of patients without diabetes :', self.data['Outcome'].value_counts()[0])
        print()
        print('Percentage of patients with diabetes :', round(self.data['Outcome'].value_counts()[1] / len(self.data) * 100, 2), '%')
        print('Percentage of patients without diabetes :', round(self.data['Outcome'].value_counts()[0] / len(self.data) * 100, 2), '%')
        print()
        print(self.data.groupby('Outcome').mean())
        print()
    
    def data_preprocessing(self):
        # Separate features (X) and target (y)
        self.X = self.data.drop('Outcome', axis=1)  # When dropping a column, set axis = 1. When dropping a row, set axis = 0.
        self.y = self.data['Outcome']
        print('Features')
        print(self.X)
        print()

        # Normalize / standardise the data to bring all column values in the same range
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        print('Normalized / Standardised features')
        print(self.X)
        print()

        # Split the data into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)
        # stratify ensures that the dataset is split in the same proportion based on the given column
        # random_state ensures that the dataset is split in a certain way, different values change the way the dataset is split
        print('Training set shape :', self.X_train.shape, self.y_train.shape)
        print('Test set shape :', self.X_test.shape, self.y_test.shape)
        print()
    
    def model_training(self):
        # Training a support vector machine classifier
        self.model.fit(self.X_train, self.y_train)
        print('Model trained successfully.')
        print()
    
    def model_evaluation(self):
        # Evaluating the model on the training set
        training_pred = self.model.predict(self.X_train)
        training_accuracy = accuracy_score(self.y_train, training_pred)
        print('Accuracy of the model on the training set :', training_accuracy)
        print()
        
        # Evaluating the model on the test set
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Accuracy of the model on the test set :', accuracy)
        print()
        print('Confusion matrix :')
        print(pd.crosstab(self.y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
        print()
        print('Classification report :')
        print(classification_report(self.y_test, y_pred))
        print()
        print('Precision, Recall, F1-score and Support for each class :')
        print(precision_recall_fscore_support(self.y_test, y_pred))
        print()

if __name__ == '__main__':
    diabetes_prediction = Diabetes_Prediction()
    diabetes_prediction.data_analysis()
    diabetes_prediction.data_preprocessing()
    diabetes_prediction.model_training()
    diabetes_prediction.model_evaluation()