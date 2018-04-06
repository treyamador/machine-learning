# a series of functions for linear regression of housing data
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import linalg

    
def clone(*args):
    return tuple(deepcopy(x) for x in args)


def println(*args):
    print()
    for arg in args:
        print(arg)
    print()


def split_data():
    data, target = datasets.load_boston(True)
    train_data,test_data,train_target,test_target = train_test_split(
        data,(target[:, np.newaxis]),test_size=0.2,random_state=42)
    return train_data,test_data,train_target,test_target


def insert_weight(data):
    return np.insert(data,0,[1 for x in range(data.shape[0])],axis=1)


def plot_one(ax,target,predict):
    ax.scatter(target,predict,edgecolors=(0,0,0))
    data_min,data_max = np.amin(target),np.amax(target)
    ax.plot([data_min,data_max],[data_min,data_max],'k--',lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')


def plot(target,*predictions):
    fig, ax = plt.subplots(len(predictions), sharex=True)
    for index,predict in enumerate(predictions):
        plot_one(ax[index],target,predict)
    plt.show()
    

def linear_regression_scikit(train_data,test_data,train_target,test_target):
    regr = linear_model.LinearRegression()
    regr.fit(train_data,train_target)
    predict = regr.predict(test_data)
    mse = mean_squared_error(test_target,predict)
    return regr.coef_,predict,mse


def calculate_decomposition(t,X,w):
    return t - np.matmul(X,w)


def calculate_mean_squared_error(n,A,B):
    return (1/n)*np.matmul(A,B)


# TODO break this function up into smaller, reusable chunks if possible
def linear_regression_analytical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    inver = linalg.inv(np.matmul(train_data_w.T,train_data_w))
    weights = np.matmul(np.matmul(inver,train_data_w.T),train_target)
    predict = np.matmul(test_data_w,weights)
    decomp = calculate_decomposition(test_target,test_data_w,weights)
    mse = calculate_mean_squared_error(test_data.shape[0],decomp.T,decomp)
    return weights.T,predict,mse[0][0]


def calculate_gradient(data,target,weights):
    decomp = calculate_decomposition(target,data,weights)
    return calculate_mean_squared_error(data.shape[0],data.T,decomp)


def linear_regression_numerical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    weights = np.full((train_data_w.shape[1],1),3.0)
    epsilon = 0.000006476
    rounds = 500000
    for _ in range(rounds):
        gradient = calculate_gradient(train_data_w,train_target,weights)
        weights += gradient * epsilon
    predict = np.matmul(test_data_w,weights)
    decomp = calculate_decomposition(test_target,test_data_w,weights)
    mse = calculate_mean_squared_error(test_data.shape[0],decomp.T,decomp)
    return weights.T,predict,mse[0][0]


def driver():
    train_data,test_data,train_target,test_target = split_data()
    weight_sk,predict_sk,mse_sk = linear_regression_scikit(train_data,test_data,train_target,test_target)
    weight_a,predict_a,mse_a = linear_regression_analytical(train_data,test_data,train_target,test_target)
    weight_n,predict_n,mse_n = linear_regression_numerical(train_data,test_data,train_target,test_target)
    print(
        'sklearn linear regression\n',weight_sk,mse_sk,'\n\n',
        'Analyitical Linear Regression\n',weight_a,mse_a,'\n\n',
        'Numerical Linear Regression',weight_n,mse_n,'\n\n')
    plot(test_target,predict_sk,predict_a,predict_n)


if __name__ == '__main__':
    driver()
    

# end of file
