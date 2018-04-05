# a series of functions for linear regression of housing data
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import linalg

from time import time




def lin_reg_ex():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    #diabetes = datasets.load_diabetes()
    
    # for testing
    diabetes = datasets.load_boston()


    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    
def clone(*args):
    return tuple(deepcopy(x) for x in args)


def println(*args):
    for arg in args:
        print(arg)


def split_data():
    data, target = datasets.load_boston(True)
    train_data,test_data,train_target,test_target = train_test_split(
        data,(target[:, np.newaxis]),test_size=0.2,random_state=42)
    return train_data,test_data,train_target,test_target


def insert_weight(data):
    return np.insert(data,0,[1 for x in range(data.shape[0])],axis=1)


def plotter(x_test,y_test,y_pred):
    y_min,y_max = np.amin(y_test),np.amax(y_test)
    fig,ax = plt.subplots()
    ax.scatter(y_test,y_pred,edgecolors=(0,0,0))
    ax.plot([y_min,y_max],[y_min,y_max],'k--',lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def plot(x_test,y_test,y_pred):
    plt.scatter(x_test,y_test,color='blue',edgecolor='black')
    plt.plot(x_test,y_pred,color='red',linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def linear_regression_scikit(train_data,test_data,train_target,test_target):
    regr = linear_model.LinearRegression()
    regr.fit(train_data,train_target)
    target_predict = regr.predict(test_data)
    mse = mean_squared_error(test_target,target_predict)
    return target_predict,mse


def calculate_decomposition(t,X,w):
    return t - np.matmul(X,w)


def calculate_mean_squared_error(n,A,B):
    return (1/n)*np.matmul(A,B)


# TODO break this function up into smaller, reusable chunks if possible
def linear_regression_analytical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    inver = linalg.inv(np.matmul(train_data_w.T,train_data_w))
    mtx_tar = np.matmul(inver,train_data_w.T)
    weights = np.matmul(mtx_tar,train_target)
    decomp = calculate_decomposition(test_target,test_data_w,weights)
    mse = calculate_mean_squared_error(test_data.shape[0],decomp.T,decomp)
    return weights,mse[0][0]




def calculate_gradient(data,target,weights):
    decomp = calculate_decomposition(target,data,weights)
    return calculate_mean_squared_error(data.shape[0],data.T,decomp)





def linear_regression_numerical(train_data,test_data,train_target,test_target):

    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)

    weights = np.full((train_data_w.shape[1],1),1)
    epsilon = 0.01

    mse_w = deepcopy(weights)
    mse = 1000000


    for i in range(100):
        gradient = calculate_gradient(train_data_w,train_target,weights)
        weights = weights - epsilon*gradient
        
        decomp = calculate_decomposition(test_target,test_data_w,weights)
        mse = calculate_mean_squared_error(test_data_w.shape[0],decomp.T,decomp)

        println(mse)

        #println(weights.T,gradient.T,weight_candidate.T)




'''

def linear_regression_numerical(train_data,test_data,train_target,test_target):

    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)

    # todo make iterations modular?
    weights = np.full((train_data_w.shape[1],1),1)
    mse = 10000000
    epsilon = 0.01
    for i in range(1,10):
        decomp = calculate_decomposition(train_target,train_data_w,weights)
        gradient = calculate_mean_squared_error(train_data.shape[0],train_data_w.T,decomp)
        candidate_w = weights - (epsilon*i)*gradient
        pred_target = np.matmul(test_data_w,candidate_w)
        candidate_mse = calculate_mean_squared_error(test_data.shape[0],pred_target.T,pred_target)[0][0]

        #print(candidate_mse)
        println(candidate_w.T)

        if candidate_mse < mse:
            mse = candidate_mse
            weights = candidate_w
            print('true!')

        print(str(i)+':',candidate_mse,mse)


        #println('\n\n\n',train_target.shape,train_data_w.shape,weights.shape,decomp.shape,gradient.shape,candidate_w.shape)


'''


def driver():
    train_data,test_data,train_target,test_target = split_data()
    prediction,mse = linear_regression_scikit(train_data,test_data,train_target,test_target)
    weights,l_e = linear_regression_analytical(train_data,test_data,train_target,test_target)

    #println(mse,l_e)
    linear_regression_numerical(train_data,test_data,train_target,test_target)


if __name__ == '__main__':
    driver()
    

# end of file
