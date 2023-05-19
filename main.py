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

# need to create a feature selection function after data cleaning to extract the important feature
def feature_selection(dataset,target_column,method='forward'):
    # Step 1: Remove duplicate records
    dataset=dataset.drop_duplicates()
    #Step 2: Remove the constant columns
    #dataset=dataset.dropna(axis=1,how='all')
    dataset=dataset.fillna(0)
    # Step 3: Perform data transformation,if needed
    
    # Step 4: Select numeric variables
    numeric_cols=dataset.select_dtypes(include=np.number).columns.tolist()
    # Step 5: Perform feature selection
    selected_features=[]
    remaining_features=numeric_cols.copy()
    current_score=0
    best_new_score=0

    # taking care of the remaining feature, use them for model prediction
    while remaining_features:
        scores=[]
        #creating for loop to loop over the feature
        for feature in remaining_features:
            selected_features.append(feature)
            # setting up the predictor(independent variable)
            X=dataset[selected_features]
            # setting up the response(dependent varibale)
            y=dataset[target_column]

            # We need to fit the machine learning model and calculated the score (R-squared,accuracy)
            # using linear regression
            score=linear_regression(X,y)
            scores.append(score)
            # we need to remove the not use feature
            selected_features.remove(feature)
            # we are finding what are our best new feature
        best_new_feature=remaining_features[np.argmax(scores)]
            # getting the best new score
        best_new_score=np.max(scores)

    # we are trying to use either forward or backward selection
        if method=='forward':
            if best_new_score>current_score:
                selected_features.append(best_new_feature)
                remaining_features.remove(best_new_feature)
                # assighing the best new score to be the current score
                current_score=best_new_score
            else:
                break
    # then we try the backward methods
        elif method=='backward':
            if best_new_score>=current_score:
                selected_features.append(best_new_feature)
                remaining_features.remove(best_new_feature)
                current_score=best_new_score
            else:
                break
    return selected_features



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
# define another function to do ridge regression fitting
def ridge_regression(X,y,alpha):
    # Add a column of ones to the feature matrix X using np.column_stack to account for the intercept term in the linear regression equation.
    X=np.column_stack((np.ones_like(y),X))
    #Calcuated the coefficients using the ridge regression equation
    XTX=X.T.dot(X)
    #taking care of the identity matrix
    I=np.eye(XTX.shape[0])
    # getting the parameter theta
    theta=np.linalg.inv(XTX+alpha*I).dot(X.T).dot(y)
    # next, we are predicting the values by multiplying the feature matrix
    y_pred=X.dot(theta)
    #Return the predicted values
    return y_pred
    
# def polyfit2d(x,y,m):
#     order=int(np.sqrt(len(m)))-1
    
#     #using itertools

#     ij=itertools.product(range(order+1),range(order+1))

#     z=np.zeros_like(x)

#     for a,(i,j) in zip(m,ij):
#         z+=a*x**i*y**j

#     return z


































# entry point of the function
if __name__=="__main__":

    #get_dataset()
    df=get_dataset()
    df=clean_dataset(df)
    # we wish to only check the numeric columns correlation, we wish to drop the catgorical columns
    df_numeric=df.select_dtypes(include=np.number)
    #before we do the feature seection, we need to chck for mutlicolinearlity
    # first we need to get the correlation matrix
    corr_matrix=df_numeric.corr()
    # next, we wish to vosualized the correlation matrix
    plt.figure(figsize=(10,8))
    cmap=cm.get_cmap('RdBu',30)
    # then we wish to use imshow
    plt.imshow(corr_matrix,interpolation='nearest',cmap=cmap)
    # take care of the colorbar
    plt.colorbar()
    #setting up the tickmark
    tick_marks=np.arange(len(corr_matrix.columns))
    #setting up the xticks
    plt.xticks(tick_marks,corr_matrix.columns,rotation=90)
    # setting up the yticks
    plt.yticks(tick_marks,corr_matrix.columns)
    # we wish to show the graph
    plt.show()
    # getting the selected features
    selected_features=feature_selection(df,'price',method='forward')
    # print out the selected features
    print(selected_features)
    # intialized the X variable(predictor variable)
    #X=df[['mileage']].fillna(0)
    # adding another independent varibale Sold_Year
    # decalre the target variable
    #X=df[['mileage','Sold_Year']].fillna(0)
    #y=df['price'].fillna(0)
    #linear_regression_coefficients=linear_regression(X,y)
    # we are checking the regression coefficients
    #print(linear_regression_coefficients)
    # then, we wish to Calculate predictions
    #y_pred=linear_regression(X,y)
    # next, we wish to Calculated prediction using polynominal fit
    
    # next, we wish to do a polynominal fit
    #x=df['mileage'].fillna(0)
    #y=df['price'].fillna(0)
    # decalre the order variable
    #Specify the desired order of the polynomial
    #order=2
    # getting a try block
    # try:
    #     # Perform polynominal fitting
    #     polyfit_coefficients=np.polyfit(x,y,order)
        #m=polyfit2d_coefficients

    #     # Calling the polyfit2d function
    #     polyfit_result=polyfit2d(x,y,polyfit_coefficients)
    #     # lastl, we want to print out the result
    #     print(polyfit_result)
    #     y_pred1=polyfit2d(x,y,order)
    #     # we need to Calculated R-squared
        ## Sum of squared residuals
    # ssr=np.sum((y-y_pred)**2)
    #     # Total sum of squares
    # sst=np.sum((y-np.mean(y))**2)
    #     # Calcualating the R square
    # r2=1-(ssr/sst)
    #     # we wish to prit out r2
    # print("R-squared",r2)
    #     # next, we wish to Calculated Mean Squared Error(MSE)
    # mse=np.mean((y-y_pred)**2)
    # print("Mean Squared Error(MSE):",mse)

    #In this modified function, an additional parameter alpha is introduced, which controls the regularization strength. Higher values of alpha result in stronger regularization. Ridge regression adds the L2 regularization term (alpha * I) to the normal equation, where I is the identity matrix.To use the ridge_regression function, you need to provide the feature matrix X, the target variable y, and the value of alpha. Here's an example:
    # checking the result of ridge regression
    #X=df[['mileage']].fillna(0)
    #y=df['price'].fillna(0)
    #alpha=0.1
    # getting the ridge_regression_coefficients
    #ridge_regression_coefficients=ridge_regression(X,y,alpha)
    # we wish to print out the coefficients
    #print(ridge_regression_coefficients)
    # then, we wish to Calculate predictions
    #y_pred2=linear_regression(X,y)
    ## Sum of squared residuals
    #ssr=np.sum((y-y_pred2)**2)
        # Total sum of squares
    #sst=np.sum((y-np.mean(y))**2)
        # Calcualating the R square
    #r2=1-(ssr/sst)
        # we wish to prit out r2
    #print("R-squared",r2)
        # next, we wish to Calculated Mean Squared Error(MSE)
    #mse=np.mean((y-y_pred2)**2)
    #print("Mean Squared Error(MSE):",mse)

    # except np.linalg.LinAlgError as e:
    #     print("Error in linear least squares computation",e)
    #polyfit2d_coefficients=polyfit2d(df['mileage'],df['Sold_Year'],df['price'])
    #print(polyfit2d_coefficients)
    print(df.dtypes)
    #car_information_entering()
    #print(mileage,Car_Model,zipcode,operation,Sold_Year)










#     In addition to forward and backward feature selection, there are other methods for choosing features in machine learning. Some commonly used methods include:

# Recursive Feature Elimination (RFE): RFE is an iterative method that starts with all features and gradually eliminates the least important features based on a specified model's performance. It recursively fits the model and removes the least important features until a desired number of features remains.

# L1 Regularization (Lasso): L1 regularization adds a penalty term to the model's cost function, which encourages sparsity by shrinking the coefficients of less important features to zero. This allows for feature selection as features with zero coefficients are excluded from the model.

# Tree-based Methods: Decision trees and tree-based ensemble models such as Random Forest and Gradient Boosting can provide insights into feature importance. By analyzing the importance scores assigned to each feature, you can select the most relevant ones.

# Univariate Selection: Univariate feature selection methods evaluate each feature independently based on statistical tests or scoring functions. Features with the highest scores or p-values below a specified threshold are selected.

# Principal Component Analysis (PCA): PCA is a dimensionality reduction technique that transforms the original features into a new set of uncorrelated variables called principal components. The principal components with the highest variance can be selected as the most informative features.

# Correlation Matrix: You can compute the correlation matrix between features and the target variable. Features with high correlation (either positive or negative) with the target can be selected as relevant features.

# Mutual Information: Mutual information measures the dependence between two variables, taking into account both linear and non-linear relationships. Features with high mutual information with the target variable can be considered relevant.

# These are just a few examples of feature selection methods. The choice of method depends on the specific problem, dataset characteristics, and the model you are using. It's often beneficial to try different methods and compare their performance to choose the most appropriate features for your machine learning task.