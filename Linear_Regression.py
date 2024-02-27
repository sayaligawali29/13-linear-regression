# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:34:54 2024

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
wcat=pd.read_csv('wc-at.csv')
#EDA
#measures the central tendancy
#measures od dispersion 
#third moment business decision
#fourth moment business decision
wcat.info()
wcat.describe()
#Graphical Representation
import matplotlib.pyplot as plt
#plotting bar graph
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
sns.displot(wcat.AT)
#Data is right skewed
#Scatter plot
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
#direction:positive ,linearity:moderate,strength:poor
#now let us calculate correlation coeffient
np.corrcoef(wcat.Waist,wcat.AT)
#Let us check the direction using cover factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#now let us apply to linear regression model
import statsmodels.formula.api as smf
#All machine learning algorithm are implemented using sklearn
#but for this statemodel
#package is being used because it gives you backend calculations of
#bita=0 and bits-1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()
#OlS helps to find best fit model,which causes 
#least square error
# first you check R squared value=0.670,if  R square=0.8
#means that model is best fit
#fit,if R-square=0.8 to 0.6 moderate fit
#next you check  P>|t|=0,it means less than alpha
#alpha is 0.05 hence the model is accepted

#Regressiom Line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,pred1,"r")
plt.show()

#error calculations
res1=wcat.AT-pred1
np.mean(res1)
#It must be zero and here it 10^14=-0
res_sqr1=res1*res1
msel=np.mean(res_sqr1)
rmsel=np.sqrt(msel)
rmsel
#32.76.76 lesser the value better the model
#how to improve this model transformation of
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#r value is 0.82<0.85 hence moderate linearity
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()
#Again check the R square value=0.67 which is less than 0.8
#p value is 0 less than 0.05
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
#check wcat and pred2 from variable explorer
#scatterdiagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['predicted line','observed data'])
#Error calculation
res2=wcat.AT-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#there no considerable changes
#now let us change y value instead of x
plt.scatter(x=wcat['Waist'],y=np.log(wcat['AT']),color='orange')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
# R VALUE is 0.84<0.85 hence moderate linearity
model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary
#Again check the R square value=0.707 which is less than 0.8
#p value is 0.02 less than 0.05
pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)
#check wcat and pred3 from variable explorer
#scatterdiagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred3,'r')
plt.legend(['predicted line','observed data'])
#Error calculation
res3=wcat.AT-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#RMSE is 38.53
#Polynomial transformation
#x=Waist,x^2=Waist*Waist,y=log(at)
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and X=Waist
model4.summary()
#R-Squarres=0.779<0.85,there is scope of improvement
#p=0.000<0.05 hence acceptable
#bita-0=-7.8241
#bita-1=0.2289
pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at
########################
#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted Line','Observed data_model3'])
plt.show()
###############################
#Error calculations
res4=wcat.AI-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#32.24
#Among all the models model4 is the best
###########################
data={"model":pd.Series(["SLR","log_model","Exp_model","Poly_model"])}
datatable_rmse=pd.DataFrame(data)
table_rmse=pd.DataFrame(data)
table_rmse
###################################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and X=Waist
final_model.summary()
#R-squares=0.779<0.85, there is scope of improvement
#p=0.000<0.05 hence acceptable
#bita-0=7.8241
#bita-1=0.2289
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at
###########################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at
############################
#Evaluation on test data
test_err=test.AT-test_pred_at
test_sqr=test_err*test_err
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
############################
#Evalution on train data
train_res=test.AT-test_pred_at
train_sqr=test_err*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#############################
#test_rmse>train mse