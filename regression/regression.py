# a series of functions for linear regression of housing data


# import numpy for use of matrix multiplication
import numpy as np

# import scipy for use in linear regression base testing
from scipy import linalg

# allows a deep copy
from copy import deepcopy

# import sklearn modules for use in lin regression
# import function to split data
from sklearn.model_selection import train_test_split

# import function to generate mean squared error
from sklearn.metrics import mean_squared_error

# import dataset for use in boston dataset
from sklearn import datasets, linear_model

# import matplotlib to allow visualization of data
import matplotlib.pyplot as plt

# import LineCollection to draw lines
from matplotlib.collections import LineCollection


def clone(*args):
    ''' function to deepcopy and return a series of arguments '''
    return tuple(deepcopy(x) for x in args)


def println(*args):
    ''' print several arguments, each on a newline '''
    # start on new line
    print()
    # iterate through input arguments
    for arg in args:
        # print argument on new line
        print(arg)
    # move onto a new line
    print()


def split_boston_data():
    ''' load and split boston data '''
    # load data from boston dataset
    data, target = datasets.load_boston(True)
    # break up data into 80% training and 20% testing
    # with input data and output targets
    train_data,test_data,train_target,test_target = train_test_split(
        data,(target[:, np.newaxis]),test_size=0.2,random_state=42)
    # return training and testing data and target
    return train_data,test_data,train_target,test_target


def plot(plot_type,target,*predictions):
    ''' plot a number of predictions  '''
    # create number of subplots based on number of predictions
    fig, ax = plt.subplots(len(predictions), sharex=True)
    # iterate through each prediction
    for index,predict in enumerate(predictions):
        # plot single subplots
        #plot_dots(ax[index],target,predict)
        plot_type(ax[index],target,predict)
    # show the multiple subplots
    fig.show()
    


def plot_lines(ax,target,predict):
    # plot a single subplot based on targets and predictions
    
    # create single scatterplot
    ax.scatter(target,predict,edgecolors=(0,0,0))
    #ax.plot(target,predict,'r.',markersize=12)

    #points = [[[i, target[i]], [i, predict[i]]] for i in range(min(len(target),len(predict)))]
    
    #lines = LineCollection(lines, array=z, cmap=plt.cm.rainbow, linewidths=5)
    #ax.add_collection(lines)
    
    #ax.plot(points,edgecolors=(0,0,0))
    #lines = LineCollection(points,linewidths=1,colors='green')
    #ax.add_collection(lines)

    data_min,data_max = np.amin(target),np.amax(target)
    ax.plot([data_min,data_max],[data_min,data_max],'k.--',lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')


def plot_dots(ax,target,predict):
    ''' plot a single subplot based on targets and predictions '''

    N = len(predict)
    x = np.arange(N)
    inds = np.argsort(target.T[0])
    targ_s = target.T[0][inds]
    pred_s = predict.T[0][inds]

    points = np.array([[[i, targ_s[i]], [i, pred_s[i]]] for i in range(N)])
    lines = LineCollection(points,linewidths=1,colors='black',zorder=0)
    lines.set_linewidths(0.5*np.ones(N))

    ax.plot(x,targ_s,'r-',linewidth=2)
    ax.plot(x,pred_s,'g.',markersize=2)
    ax.add_collection(lines)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')


def print_one(title,weight,mse_train,mse_test):
    print(title,'\n')
    print('Train Mean Squared Error:',mse_train,'\n')
    print('Test Mean Squared Error:',mse_test,'\n')
    weight_str = [str(w)+'x'+str(i) for i,w in enumerate(weight[0])]
    print('Weights:\n',' + '.join(weight_str),'\n\n')


def print_summary(*args):
    for i,c in enumerate(['sklearn','Analytical','Numerical']):
        off = i*3
        print_one(c,args[off],args[off+1],args[off+2])


def insert_weight(data):
    return np.insert(data,0,[1 for x in range(data.shape[0])],axis=1)


def calculate_decomposition(t,X,w):
    return t - np.matmul(X,w)


def calculate_mean_squared_error(data,target,weights):
    decomp = calculate_decomposition(target,data,weights)
    mse = (1/data.shape[0])*np.matmul(decomp.T,decomp)
    return mse[0][0]


def calculate_gradient(data,target,weights):
    decomp = calculate_decomposition(target,data,weights)
    return (1/data.shape[0])*np.matmul(data.T,decomp)


def calculate_prediction(data,weights):
    return np.matmul(data,weights)


def linear_regression_scikit(train_data,test_data,train_target,test_target):
    regr = linear_model.LinearRegression()
    regr.fit(train_data,train_target)
    mse_target = mean_squared_error(train_target,regr.predict(train_data))
    predict = regr.predict(test_data)
    mse_test = mean_squared_error(test_target,predict)
    return regr.coef_,predict,mse_test,mse_target


def linear_regression_analytical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    inver = linalg.inv(np.matmul(train_data_w.T,train_data_w))
    weights = np.matmul(np.matmul(inver,train_data_w.T),train_target)
    predict = calculate_prediction(test_data_w,weights)
    mse_test = calculate_mean_squared_error(test_data_w,test_target,weights)
    mse_target = calculate_mean_squared_error(train_data_w,train_target,weights)
    return weights.T,predict,mse_test,mse_target


def linear_regression_numerical(train_data,test_data,train_target,test_target):
    train_data_w = insert_weight(train_data)
    test_data_w = insert_weight(test_data)
    weights = np.full((train_data_w.shape[1],1),0.0)
    epsilon = 0.000006476
    rounds = 500000

    rounds = 10000

    for _ in range(rounds):
        gradient = calculate_gradient(train_data_w,train_target,weights)
        weights += gradient * epsilon
    predict = calculate_prediction(test_data_w,weights)
    mse_test = calculate_mean_squared_error(test_data_w,test_target,weights)
    mse_train = calculate_mean_squared_error(train_data_w,train_target,weights)
    return weights.T,predict,mse_test,mse_train


def driver():
    train_data,test_data,train_target,test_target = split_boston_data()    
    weight_sk,predict_sk,mse_test_sk,mse_train_sk = \
        linear_regression_scikit(
            train_data,test_data,train_target,test_target)
    weight_a,predict_a,mse_test_a,mse_train_a = \
        linear_regression_analytical(
            train_data,test_data,train_target,test_target)
    weight_n,predict_n,mse_test_n,mse_train_n = \
        linear_regression_numerical(
            train_data,test_data,train_target,test_target)
    print_summary(
        weight_sk,mse_test_sk,mse_train_sk,
        weight_a,mse_test_a,mse_train_a,
        weight_n,mse_test_n,mse_train_n)
    plot(plot_dots,test_target,predict_sk,predict_a,predict_n)
    plot(plot_lines,test_target,predict_sk,predict_a,predict_n)
    plt.show()


if __name__ == '__main__':
    ''' entry point of program '''
    driver()


# end of file

