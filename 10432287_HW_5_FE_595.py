
# coding: utf-8

# ## Boston House price prediction

# ### Loading the modules and the dataset



from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_boston
boston_data=load_boston()
from pylab import savefig


boston_data.data.shape



print(boston_data.keys())


boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston.head()

boston['MEDV'] = boston_data.target

boston.isnull().sum()


# ### Performing EDA on the data

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)


# ### From the correlation plot it is observed that the target variable MEDV has strong positive relation with RM variable and strong negative relation with LSTAT variable

correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
svm = sns.heatmap(data=correlation_matrix, annot=True)


figure = svm.get_figure()    
figure.savefig('correlation_matrix.png', dpi=400)


# ### Price tend to decrease as LSTAT value decreases and MEDV value increases as the RM values increases

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

# ### using the high negative and postive independent variables for Linear Regression 

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### Linear Regression model and accuracy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

### testing the error ana

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("The model performance for training set")
print('RMSE is {}'.format(rmse))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))


print("The model performance for testing set")
print('RMSE is {}'.format(rmse))



#plt.plot(range(10))
plt.savefig('finalplot.png')
