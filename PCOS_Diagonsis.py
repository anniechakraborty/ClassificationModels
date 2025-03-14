import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class PCOS_Diagnosis:
    # Data Source: Kaggle (https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset)
    data = pd.read_csv('data/pcos_dataset.csv')
    X=[]
    y=[]

    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]

    # creating a logistic regression model
    model = LogisticRegression()

    def data_description(self):
        print(type(self.data))
        # dimensions
        print(self.data.shape)
        print()

        # top 5 data cases 
        print(self.data.head())
        print()

        # statistical description of the data
        print(self.data.describe())
        print()

        # number of data cases for negative diagosis (0) and psoitive diagnosis (1)
        print(self.data.iloc[:,-1].value_counts())
        print()

        # mean values for all the attributes over the positive and negative diagonsis
        print(self.data.groupby('PCOS_Diagnosis').mean())
        print()
    
    def data_preprocessing(self):
        # check for missing values
        print(self.data.isnull().sum())
        print()

        # replace missing values with the mean of the respective column
        self.data.fillna(self.data.mean(), inplace=True)

        # separating data and label
        self.X = self.data.drop('PCOS_Diagnosis', axis=1) # axis removes the rows
        self.y = self.data['PCOS_Diagnosis']

        print(self.X)
        print()
        print(self.y)
        print()

        # Normalize / standardise the data to bring all column values in the same range
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        print('Normalized / Standardised features')
        print(self.X)
        print()

        # splitting data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=24)
        # the stratify parameter is used to consider equal amount of data cases from each label during the split

        print(self.X_test.shape, self.X_train.shape)
        print(self.y_test.shape, self.y_train.shape)
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

        if prediction[0] == 0:
            print("The person does not have PCOS")
        else:
            print("The person has PCOS")
        print()


if __name__ == "__main__":
    diagnosis = PCOS_Diagnosis()
    diagnosis.data_description()
    diagnosis.data_preprocessing()
    diagnosis.model_training()
    diagnosis.model_evaluation()

    # Making predictions
    feature_names = ["Age", "BMI", "Menstrual_Irregularity", "Testosterone_Level(ng/dL)", "Antral_Follicle_Count"]
    pos_input = np.asarray((29,29.7,1,98.7,14))
    diagnosis.prediction(pd.DataFrame([pos_input], columns=feature_names))

    neg_input = np.asarray((38,24.2,1,92.5,10))
    diagnosis.prediction(pd.DataFrame([neg_input], columns=feature_names))