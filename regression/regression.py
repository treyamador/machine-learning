# a series of functions for linear regression of housing data
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import linalg



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


def println(*args):
    for arg in args:
        print(arg)


def clone(*args):
    return tuple(deepcopy(x) for x in args)


def split_data():
    data, target = datasets.load_boston(True)
    train_data,test_data,train_target,test_target = train_test_split(
        data,(target[:, np.newaxis]),test_size=0.2,random_state=42)
    return train_data,test_data,train_target,test_target


def load_data(data_set):
    data = data_set()
    x_axis = data.data[:,np.newaxis,2]
    x_train = x_axis[:-20]
    x_test = x_axis[-20:]
    y_train = data.target[:-20]
    y_test = data.target[-20:]
    return x_train,x_test,y_train,y_test



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


def lin_regr_sklearn(x_train,x_test,y_train,y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return y_pred,mse,r2


def lin_regr_analytical(x_train,x_test,y_train,y_test):
    y_yt = np.matmul(y_train.T,y_train)
    print(y_yt)
    return
    mtx_inv = linalg.inv(y_yt)
    mtx_p = np.dot(mtx_inv,y_train.T)
    weights = np.dot(mtx_p,x_train)
    return weights


def linear_regression_scikit(train_data,test_data,train_target,test_target):
    regr = linear_model.LinearRegression()
    regr.fit(train_data,train_target)
    target_predict = regr.predict(test_data)
    mse = mean_squared_error(test_target,target_predict)
    return target_predict,mse


'''

def linear_regression_analytical(train_data,test_data,train_target,test_target):
    inver = linalg.inv(np.matmul(train_data.T,train_data))
    mtx_tar = np.matmul(inver,train_data.T)
    weights = np.matmul(mtx_tar,train_target)
    decomp = test_target - np.matmul(test_data,weights)
    mse = (1/(test_data.shape[0]))*np.matmul(decomp.T,decomp)
    return weights,mse

'''


def insert_weight(data):
    return np.insert(data,0,[1 for x in range(data.shape[0])],axis=1)


def linear_regression_analytical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    inver = linalg.inv(np.matmul(train_data_w.T,train_data_w))
    mtx_tar = np.matmul(inver,train_data_w.T)
    weights = np.matmul(mtx_tar,train_target)
    decomp = test_target - np.matmul(test_data_w,weights)
    mse = (1/(test_data.shape[0]))*np.matmul(decomp.T,decomp)
    return weights,mse[0][0]



def linear_regression_numerical(train_data,test_data,train_target,test_target):
    pass





def driver():
    train_data,test_data,train_target,test_target = split_data()
    prediction,mse = linear_regression_scikit(train_data,test_data,train_target,test_target)
    weights,l_e = linear_regression_analytical(train_data,test_data,train_target,test_target)

    println(mse,l_e)


if __name__ == '__main__':
    driver()
    

# end of file
