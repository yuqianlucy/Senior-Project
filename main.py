# Importing the require library
import pandas as pd
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
#from sklearn.metrics import r2_score,mean_absolute_error
#from sklearn.linear_model import LinearRegression


# Creating a function to get the data in general
def get_dataset():
    df=pd.read_csv("C:/Users/yuqia/OneDrive/Desktop/Senior Project/_Car_Donation_Version7 - Raw Detail (2).csv")
    print(df.head())
    return df
# Creating a function to clean out dataset
#def clean_dataset(dataset=get_dataset()):
def clean_dataset(dataset):
    # step1: we need to check the duplicate inside the dataset
    
    # after checking we need to remove the duplicated record
    dataset=dataset.drop_duplicates()
    
    # step 4: we need to remove the constant columns in the dataset
    dataset=dataset.dropna(axis=1,how='all')
    # step 5: data transformation of the feature
    dataset['price']=dataset['price'].str.replace(',','')
    dataset.price=dataset.price.astype(float)
    dataset['mileage']=dataset['mileage'].str.replace(',','')
    dataset['mileage'] = pd.to_numeric(dataset['mileage'],errors='coerce')
    #dataset['mileage'] = pd.to_numeric(dataset['mileage'],errors='coerce')
    #dataset['mileage'] = pd.to_numeric(dataset['mileage'],errors='coerce')
    dataset.mileage=dataset.mileage.astype(float)
    csv_file=dataset.to_csv(os.path.join('C:/Users/yuqia/OneDrive/Desktop/Senior Project','clean_data.csv'))
    #print(dataset.duplicated())
    #return dataset.duplicated()
    print(dataset)
    return dataset



# creating a function to ask input from the user
def car_information_entering():
# declare several variables for user to enter
    mileage=int(input("Please enter the mileage of the car"))
    car_model=(input("Please enter the Model of the Car"))
    zipcode=(input("Please enter the zipcode of the car"))
    operation=(input("Please enter which operation the car is from"))
    Sold_Year=(input("Please enter the sold year of the car"))
    print(mileage,car_model,zipcode,operation,Sold_Year)
    return mileage,car_model,zipcode,operation,Sold_Year

#def model_selecting(data1,feature=car_information_entering()):
#def polyfit2d(x,y,m,data1=get_dataset()):

#X represents the matrix of input features (each row corresponds to a data point, and each column represents a feature), and y represents the target variable.
def linear_regression(X,y):
    #intialized x variable and reshape x to a 2D array
    #Add a column of ones to the feature matrix X using np.column_stack to account for the intercept term in the linear regression equation.
    X=np.column_stack((np.ones_like(y),X))
    # need to Convert y to a 2D array
    #y=np.array(y).reshape(-1,1)
    # Calculate the coefficients using the normal equation
    #Calculate the coefficients (theta) using the normal equation: theta = (X^T * X)^-1 * X^T * y, where X^T is the transpose of X.
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    # we are predicting the values
    #Predict the values by multiplying the feature matrix X with the coefficients theta.
    y_pred=X.dot(theta)
    # we are returning the result
    return y_pred

    
def polyfit2d(x,y,m):
    order=int(np.sqrt(len(m)))-1
    
    #using itertools
    ij=itertools.product(range(order+1),range(order+1))

    z=np.zeros_like(x)

    for a,(i,j) in zip(m,ij):
        z+=a*x**i*y**j

    return z


































# entry point of the function
if __name__=="__main__":
    #get_dataset()
    df=get_dataset()
    df=clean_dataset(df)
    # intialized the X variable(predictor variable)
    #X=df[['mileage']].fillna(0)
    # adding another independent varibale Sold_Year
    # decalre the target variable
    X=df[['mileage','Sold_Year']].fillna(0)
    y=df['price'].fillna(0)
    linear_regression_coefficients=linear_regression(X,y)
    # we are checking the regression coefficients
    print(linear_regression_coefficients)
    # then, we wish to Calculate predictions
    y_pred=linear_regression(X,y)
    # we need to Calculated R-squared
    ## Sum of squared residuals
    ssr=np.sum((y-y_pred)**2)
    # Total sum of squares
    sst=np.sum((y-np.mean(y))**2)
    # Calcualating the R square
    r2=1-(ssr/sst)
    # we wish to prit out r2
    print("R-squared",r2)
    # next, we wish to Calculated Mean Squared Error(MSE)
    mse=np.mean((y-y_pred)**2)
    print("Mean Squared Error(MSE):",mse)
    #polyfit2d_coefficients=polyfit2d(df['mileage'],df['Sold_Year'],df['price'])
    #print(polyfit2d_coefficients)
    print(df.dtypes)
    #car_information_entering()
    #print(mileage,Car_Model,zipcode,operation,Sold_Year)