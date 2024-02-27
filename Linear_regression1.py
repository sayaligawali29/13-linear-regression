# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:55:57 2024

@author: user
"""
import pandas as pd
import numpy as np
import matlplotlib.pyplot as plt
import seaborn as sns
cars=pd.read_csv("cars.csv")

#EDA
#measure the central the tendency
#measure the dispertion
#third moment busness decision
##fourth moment busness decision

#probability ditribution
cars.describe()
#graphical representation
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.displot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
#There are several outliers in HP columns
#similar operations are expected for other three column
sns.displot(cars.MPG)
#data is slight left distributed
plt.boxplot(cars.MPG)
#There are no outliers 
sns.distplot(cars.VOL)
#data is slight left distributed
plt.boxplot(cars.VOL)
sns.distplot(cars.SP)
#data is slight right distributed
plt.boxplot(cars.SP)
#There are several outliers
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
#There are several outliers#Now let us plot joint plot is to show scatterplot
#Histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MGP'])

#Now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times each value occured
#92 HP value occured 7 times

##QQ Plot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist="norm",plot=pylab)
plt.show()
#MPG data is normally distributed
#There are 10 scatter plots need to be plotted,one by to plot
#so,we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])

import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#R square value observed is 0.771<0.85
#p.values of WT and VOL is0.814 and 0.556 which is very high
#it means it is greaater than 0.05,WT and VOL columns
#we need to ignore
#or delete instead deleting 81 entries
#let us check row wise outliers
#identifying is there any influential value.
#to check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to data frame and check 76th entry
#let us delete that entry
cars_new=cars.drop(cars.index[[76]])

#again apply regression to cars_new
ml_new=smf.ols('MPG~WT+HP+SP',data=cars_new).fit()
ml_new.summary()
#R-sqaure vslue is 0.819 but p values are same,hence not 
#now next 

#VOL has got -0.529 and for WT=-0.526
#WT is less hence can be deleted
#another approach is to check the collinearity
#square is giving that value
# we will have to apply regression w.r.t.x1 and input
#as x2,x3 and x4 so on so forth
rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fits().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
#VIF is variance influential factor calculating VIF helps
#of X1 w.r.t.x2,x3 and x4
rsq_WT=smf.ols("HP~WT+VOL+SP",data=cars).fit().rsquared
vif_WT=1/(1-rsq_WT)
vif_WT

rsq_VOL=smf.ols("HP~WT+VOL+SP",data=cars).fit().rsquared
vif_VOL=1/(1-rsq_VOL)
vif_VOL

rsq_SP=smf.ols("HP~WT+VOL+SP",data=cars).fit().rsquared
vif_SP=1/(1-rsq_SP)
vif_SP
#vif_wt=639.53 and vif_vol=638.80 hence vif_wt
#is greater thumb rule is vif should not be greater than 1
#Storing the values in dataframe
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_WT]}
vif_frame=pd.DataFrame(d1)
vif_frame

#let us drop WT and apply correlation to remailing three
final_ml=smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
#R sqaure is 0.770 and p values 0.00,0.012<0.05
#prediction
pred=final_ml.predict(cars)

#QQ plots 
res=final_ml.resid
sm.qqplot(res)
plt.show()
#This  QQ plot is on residual which is obtained on traing
#errors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#Let us plot the residual plot,which takes the residuals value and data
sns.residplot(x=pred,y=cars.MPG,lowers=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('FItted VS Residual')
plt.show()
#residual post

#Splitting the data into train and test data

