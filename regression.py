import matplotlib.pyplot as plt
import csv 
import numpy as np
import ast
import pandas as pd
from pylab import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt

#task 1
#read csv into pandas data frame
df = pd.read_csv("ovbarve.csv")

#function to plot histogram
def plot_histogram(array, count):
    plt.hist(array, bins=30)
    plt.ylabel('Probability')
    plt.title("Histogram :"+str(count))
    plt.xlabel('Value bins')
    plt.show()

#calculate mean and variance of each column and plot histogram
regression_variable = list(df.keys())
for count,variable in enumerate(regression_variable):
    print("mean of regression_variable: x"+str(count+1), df[variable].mean())
    print("variance of regression_variable: x"+str(count+1), df[variable].var())
    plot_histogram(df[variable], count)

#plot correlation matrix
print(df.corr())
pd.scatter_matrix(df)
plt.show()

"""
# remove outliers at 2 std deviations and outside
elements = np.array(x1)

mean = np.mean(elements, axis=0)
sd = np.std(elements, axis=0)

final_list = [x for x in x1 if (x > mean - 2 * sd)]
final_list = [x for x in final_list if (x < mean + 2 * sd)]
print(final_list)

#plot correlation matrix
df = pd.read_csv("ovbarve.csv")
print(df.corr())
pd.scatter_matrix(df)
plt.show()
"""

#Task 2
#Linear regression

regression_variables = [x1, x2, x3, x4, x5]

#reshape to fit for regression
regression_variables = [[[each_feature] for each_feature in array] for array in regression_variables]

for count,variable in enumerate(regression_variables):
    fig = plt.figure()
    X = sm.add_constant(variable) # adding a constant
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)
    
    #calculate residual array
    residual =  y - predictions  
    
    #residual scatter plot 
    plt.subplot(3, 1, 1)    
    plt.scatter(variable, residual,color='g')
    plt.ylabel('residual');
    plt.title("Residual scatter plot :"+str(count))
    plt.xlabel('x - variable')


    #residual histogram
    plt.subplot(3, 1, 2)
    plt.hist(residual, bins=30)
    plt.title("Histogram of residual:"+str(count))        

    lm=linear_model.LinearRegression()
    lm.fit(variable, y)

    plt.subplot(3, 1, 3)
    plt.scatter(variable, y,color='g')
    plt.plot(variable, lm.predict(variable),color='k')
    plt.title("regression fit:"+str(count))            
    plt.show()
    
    print('Coeff of determination:'+str(count), lm.score(variable, y))
    print('correlation is:'+str(count), math.sqrt(lm.score(variable, y)))
    
#Task 3:
#Multipvariable regression

fig = plt.figure()

multi_variables = tuple(zip(x1, x2, x3, x4, x5))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual = predictions.resid
residual =  y - predictions 
#plt.scatter(predictions, residual,color='g')
#plt.show()
sm.qqplot(residual, line='45')
plt.show()


print_model = model.summary()
print(print_model)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig.show()

input()

