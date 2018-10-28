import matplotlib.pyplot as plt
import csv 
import os
import numpy as np
import ast
import pandas as pd
from pylab import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from sklearn import *
from scipy.stats import chisquare
import scipy.stats as stats
from scipy import stats


#task 1
#read csv into pandas data frame
df = pd.read_csv("ovbarve.csv")
os.chdir("graphs")

#function to plot histogram
def plot_histogram(array, count):
    if count == 5:
        fig = plt.figure()
        plt.hist(array, bins=30)
        plt.ylabel('Probability')
        plt.title("Histogram Y")
        plt.xlabel('Dependent variable with bin size as 30')
        fig.savefig("Histogram Y.png")
    else:
        fig = plt.figure()
        plt.hist(array, bins=30)
        plt.ylabel('Probability')
        plt.xlabel('independent variables with bin size as 30')
        plt.title("Histogram X:"+str(count + 1))
        fig.savefig("Histogram X:"+str(count + 1)+".png")
    plt.show()

#calculate mean and variance of each column and plot histogram
regression_variable = list(df.keys())
for count,variable in enumerate(regression_variable):
    print("mean of regression_variable: x"+str(count+1), df[variable].mean())
    print("variance of regression_variable: x"+str(count+1), df[variable].var())
    plot_histogram(df[variable], count)


#plot correlation matrix
#fig = plt.figure()
print(df.corr())
pd.scatter_matrix(df)
plt.savefig("correlation_scatter_plot.png")
plt.show()

# remove outliers at 3 std deviations and outside
print(df.shape)

df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(df.shape)
input("press enter to continue")

#Task 2
#Linear regression

regression_variables = list(df.keys())
y = df[regression_variables[-1]]

for count,variable in enumerate(regression_variables[:-1]):

    #prepare data by reshaping vector eg. [1, 2, 3] to [[1], [2], [3]]    
    current_independent_variable = [[i] for i in df[variable]]
    
    X = sm.add_constant(current_independent_variable) # adding a constant
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)
    
    #calculate residual array
    residual =  y - predictions
    print("variance of residuals is :", np.var(residual))

    #residual qqplot
    sm.qqplot(residual)
    plt.title("QQ plot for single variable: X"+str(count))
    plt.savefig("QQ plot of residual for linear regression of: X"+str(count)+" .png")
    plt.show()
    

    #residual scatter plot 
    plt.scatter(predictions, residual,color='g')
    plt.ylabel('residual');
    plt.title("Residual scatter plot for single variable :"+str(count))
    plt.xlabel('x - current_independent_variable')
    plt.savefig("residual scatter for linear regression of: X"+str(count)+" .png")
    plt.show()

    #residual histogram

    plt.hist(residual, bins=30)
    plt.title("Histogram of residual:"+str(count))        
    plt.savefig("residual histogram for linear regression of: X"+str(count)+" .png")
    plt.show()

    #Chi square test
    print(stats.normaltest(residual))

    #Plot regression fit using sklearn fitting again (remove this) 
    lm=linear_model.LinearRegression()
    lm.fit(current_independent_variable, y)
    

    plt.scatter(current_independent_variable, y,color='g')
    plt.plot(current_independent_variable, lm.predict(current_independent_variable),color='k')
    plt.title("regression fit:"+str(count))            
    plt.savefig("Regression fit for linear regression of: X"+str(count)+" .png")
    plt.show()
    
    print('Coeff of determination:'+str(count), lm.score(current_independent_variable, y))
    print('correlation is:'+str(count), math.sqrt(lm.score(current_independent_variable, y)))

input("press enter to continue")

#Task 2: Polymnomial regression
#fit for higher order polynomial:
regression_variable = list(df.keys())
multi_variables = list(zip(df[regression_variable[0]], [i**2 for i in df[regression_variable[0]]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation
residual =  y - predictions 
print("variance of residuals is :", np.var(residual))

#residual scatter plot
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("Residual_scatter_plot_for_ploynomial.png")
plt.show()

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression X1:  y = a0 + a1*x1 + a2*(x1**2)")        
plt.savefig("residual histogram for polynomial regression of x1.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))


print_model = model.summary()
print(print_model)
#regression fit plots
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Residual scatter for polynomial regression plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("Regression_fit_polynomial.png")
fig.show()

#qq plot of residuals
sm.qqplot(residual)
plt.title("QQplot for polynomial regression plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("QQplot for polynomial.png")
plt.show()
input("press enter to continue")

#Task 3:
#Multipvariable regression

regression_variable = list(df.keys())
multi_variables = tuple(zip(df[regression_variable[0]], df[regression_variable[1]], df[regression_variable[2]], df[regression_variable[3]], df[regression_variable[4]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation and qqplot 
residual =  y - predictions
print("variance of residuals is :", np.var(residual))
#residual scatter plot 
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Residual_scatter_plot_for_multi_variable.png")
plt.show() 

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")        
plt.savefig("residual histogram for multi_variable regression.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))   

print_model = model.summary()
print(print_model)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Regression fit for multi variable regression plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Regression_fit_multi_variable.png")
fig.show()

sm.qqplot(residual)
plt.title("QQplot for multi variable regression plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("QQplot_for_multi_variable.png")
plt.show()

input("press enter to continue")


#Task 3: remove a non dependent variable x2
#Multipvariable regression

regression_variable = list(df.keys())
multi_variables = tuple(zip(df[regression_variable[0]], df[regression_variable[2]], df[regression_variable[3]], df[regression_variable[4]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation and qqplot 
residual =  y - predictions

#residual scatter plot 
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Residual_scatter_plot_for_multi_variable_withoutx2.png")
plt.show() 

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")        
plt.savefig("residual histogram for multi_variable regression_withoutx2.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))   

print_model = model.summary()
print(print_model)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Regression fit for multi variable regression plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Regression_fit_multi_variable_withoutx2.png")
fig.show()

sm.qqplot(residual)
plt.title("QQplot for multi variable regression plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("QQplot_for_multi_variable_withoutx2.png")
plt.show()

input("press enter to continue")



