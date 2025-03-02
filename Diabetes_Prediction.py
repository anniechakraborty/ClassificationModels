import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

class Diabetes_Prediction:
    # Data Source: Kaggle (https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction)
    data = pd.read_csv('data/diabetes.csv')

    def data_collection_and_analysis(self):
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

if __name__ == '__main__':
    diabetes_prediction = Diabetes_Prediction()
    diabetes_prediction.data_collection_and_analysis()