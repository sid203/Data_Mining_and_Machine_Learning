import lib1771650 as lib
import os 
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge


######################------Read Files-------#########################



files=sys.argv[1:]
train = pd.read_csv(files[0], index_col=0)							#train.csv
test = pd.read_csv(files[1], index_col=0)							#test.csv
rows=train.shape[0]										#1460 rows




#house_data has the combined test and train data with SalePrice dropped
#true_data consists of the SalePrice column which is the true set of labels
house_data,true_data=lib.combine_data(train,test)



######################------Pre processing-------#########################





#Check which columns has nans

#print(lib.checknans(house_data),'\n')								#Uncomment to see the output



#Filling empty Nan Values and empty columns
#Explore each of the categorical variables 
#and fill with most frequent value

#print(lib.cat_variables_counts('Alley',house_data),'\n')	 				#too many NaN values,replace by None
lib.cat_fillvalue('Alley',house_data,'None')							#fill with mode value i.e None
#print(lib.cat_variables_counts('FireplaceQu',house_data),'\n') 				#too many NaN values,replace by None
lib.cat_fillvalue('FireplaceQu',house_data,'None')						#fill with mode value i.e  None
#print(lib.cat_variables_counts('PoolQC',house_data),'\n') 					#too many NaN values,replace by None
lib.cat_fillvalue('PoolQC',house_data,'None')							#fill with mode value i.e  None
#print(lib.cat_variables_counts('PoolArea',house_data),'\n') 					#Since most of the PoolArea is 0, therefore there are lot of NaN values in PoolQC
#print(lib.cat_variables_counts('MiscFeature',house_data),'\n') 				#too many NaN values,replace by None
lib.cat_fillvalue('MiscFeature',house_data,'None')						#fill with mode value i.e  None		
#print(lib.cat_variables_counts('Fence',house_data),'\n') 					#too many NaN values,replace by None
lib.cat_fillvalue('Fence',house_data,'None')							#fill with mode value i.e  None


#Check total NaNs again
#print(lib.checknans(house_data),'\n')


#Filling mode values for each of the above columns with missing values

#print(lib.cat_variables_counts('Electrical',house_data),'\n') 					#SBrkr is the most frequent value
lib.cat_fillvalue('Electrical',house_data,'SBrkr')						#Filled NaNs with most frequent value
#print(lib.cat_variables_counts('MSZoning',house_data),'\n') 					#RL is most frequent
lib.cat_fillvalue('MSZoning',house_data,'RL') 							#Filled NaNs with most frequent value
#print(lib.cat_variables_counts('KitchenQual',house_data),'\n') 				#Filled NaNs with most frequent value
lib.cat_fillvalue('KitchenQual',house_data,'TA') 						#Filled NaNs with most frequent value
#print(lib.cat_variables_counts('Functional',house_data),'\n') 					#Filled NaNs with most frequent value
lib.cat_fillvalue('Functional',house_data,'Typ') 						#Filled NaNs with most frequent value
lib.cat_fillvalue('SaleType',house_data,'WD') 							#Filled NaNs with most frequent value
#print(house_data['MasVnrType'][house_data['MasVnrType'].isnull()==True],'\n')			#Replace NaN values with None
#print(lib.cat_variables_counts('MasVnrType',house_data)),'\n' 					#Fill None with NaN as it is the most common
house_data['MasVnrArea'][house_data['MasVnrArea'].isnull()==True] 				#Replace NaN values with 0
#print(lib.cat_variables_counts('MasVnrArea',house_data)) 					
#print(house_data[['MasVnrType','MasVnrArea']][house_data['MasVnrArea'].isnull()==True],'\n') 	
lib.cat_fillvalue('MasVnrType',house_data,'None')
lib.cat_fillvalue('MasVnrArea',house_data,0.0)
lib.cat_fillvalue('Utilities',house_data,'AllPub') 						


lib.cat_variables_counts('Exterior1st',house_data)
lib.cat_fillvalue('Exterior1st',house_data,'VinylSd')
del house_data['Exterior2nd'] 									#since most of values of Exterior1st and Exterior2nd are equal hence delete Exterior2nd





#LotArea/LotFrontage variables

names=house_data['LotArea'][house_data['LotFrontage'].isnull()] 				#Find out NaNs in LotFrontage
house_data['LotFrontage'][house_data['LotFrontage'].isnull()]=np.sqrt(names) 			#Fill NaNs with square root of LotArea for observations having NaN Values
#print(house_data['LotFrontage'].corr(house_data['LotArea'])) 					#Correlation increases to 0.62, hence filling squareroot of area is justified



#Basement Variables

bsmt_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
#print(house_data[bsmt_cols][house_data['BsmtQual'].isnull()])
lib.cat_fillvalue('BsmtQual',house_data,'None')
lib.cat_fillvalue('BsmtFinType1',house_data,'None')
lib.cat_fillvalue('BsmtFinType2',house_data,'None')
lib.cat_fillvalue('BsmtCond',house_data,'None')
lib.cat_fillvalue('BsmtExposure',house_data,'None')
lib.cat_fillvalue('BsmtFinSF1',house_data,0.0)
lib.cat_fillvalue('BsmtFinSF2',house_data,0.0)
lib.cat_fillvalue('BsmtUnfSF',house_data,0.0)
lib.cat_fillvalue('TotalBsmtSF',house_data,0.0)
lib.cat_fillvalue('BsmtFullBath',house_data,0.0)
lib.cat_fillvalue('BsmtHalfBath',house_data,0.0)


#Garage Variables

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
house_data[garage_cols][house_data['GarageType'].isnull()==True]
lib.cat_fillvalue('GarageType',house_data,'None')
lib.cat_fillvalue('GarageQual',house_data,'None')
lib.cat_fillvalue('GarageCond',house_data,'None')
lib.cat_fillvalue('GarageYrBlt',house_data,'None')
lib.cat_fillvalue('GarageFinish',house_data,'None')
lib.cat_fillvalue('GarageCars',house_data,0.0)
lib.cat_fillvalue('GarageArea',house_data,0.0)






######################------Feature Engineering-------#########################





#Developing Polynomial Features for highly correlated columns with SalesPrice

corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
#print(corr.SalePrice)


#Generating Polynomial Features for 5 most correlated columns

house_data["OverallQual-deg2"]=lib.genpolynomials(house_data["OverallQual"],2)
house_data["OverallQual-deg3"]=lib.genpolynomials(house_data["OverallQual"],3)
house_data["GrLivArea-deg2"]=lib.genpolynomials(house_data["GrLivArea"],2)
house_data["GrLivArea-deg3"]=lib.genpolynomials(house_data["GrLivArea"],3) 
house_data["GarageCars-deg2"]=lib.genpolynomials(house_data["GarageCars"],2)
house_data["GarageCars-deg3"]=lib.genpolynomials(house_data["GarageCars"],3)
house_data["GarageArea-deg2"]=lib.genpolynomials(house_data["GarageArea"],2)
house_data["GarageArea-deg3"]=lib.genpolynomials(house_data["GarageArea"],3)

house_data["OverallQual_GrLivArea"]=house_data['OverallQual']*house_data['GrLivArea']
house_data["OverallQual_GarageCars"]=house_data['OverallQual']*house_data['GarageCars']
house_data["GarageCars_GrLivArea"]=house_data['GarageCars']*house_data['GrLivArea']
house_data['OverallQual_GarageArea']=house_data['GarageArea']*house_data['OverallQual']
house_data['GarageCars_GarageArea']=house_data['GarageArea']*house_data['GarageCars']
house_data['GrLivArea_GarageArea']=house_data['GarageArea']*house_data['GrLivArea']
house_data["OverallQual_TotalBsmtSF"]=house_data['OverallQual']*house_data['TotalBsmtSF']
house_data["OverallQual_1stFlrSF"]=house_data['OverallQual']*house_data['1stFlrSF']
house_data["GrLivArea_TotalBsmtSF"]=house_data['GrLivArea']*house_data['TotalBsmtSF']
house_data["GrLivArea_1stFlrSF"]=house_data['GrLivArea']*house_data['1stFlrSF']
house_data["GarageCars_1stFlrSF"]=house_data['GarageCars']*house_data['1stFlrSF']
house_data["GarageCars_TotalBsmtSF"]=house_data['GarageCars']*house_data['TotalBsmtSF']
house_data["GarageArea_1stFlrSF"]=house_data['GarageArea']*house_data['1stFlrSF']
house_data["GarageArea_TotalBsmtSF"]=house_data['GarageArea']*house_data['TotalBsmtSF']




#Encoding of categorical variables
house_data_encoded=pd.get_dummies(house_data)






######################------Normalization-------#########################





#Normalize data 
Norm_data=np.log1p(house_data_encoded)								#Log(1+p) normalization
Norm_data_true=np.log1p(true_data)   								#Normalized labels i.e. SalesPrice


#Split into Training set and Test set
Train_data=Norm_data[:rows]
Test_data=Norm_data[rows:]







######---Regression with Regularization (or ridge regression)--###########



#Ridge regression ( or regularization)
#Returns r2 score and rmse for different values of alpha

alphas=[0.001, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 4, 5, 10, 15, 30, 50]


r2score,rmse=lib.ridge_analysis(alphas,Train_data,Norm_data_true,true_data)



#Print RMSE as a function of alphas

#lib.print_rmse_curve(rmse,alphas)

#Selecting alpha = 5 for performing ridge regression


rr=Ridge(5,fit_intercept=True,max_iter=10000)
rr.fit(Train_data.values, Norm_data_true.values)
rr.coef_
rrpreds = pd.DataFrame({"SalePrice":rr.predict(Test_data)}, index=Test_data.index)

#De-Normalize predictions
rrpreds.SalePrice=np.exp(rrpreds.SalePrice)


#Save predictions
rrpreds.to_csv("pred.csv")


#Generate Plots and figures

#lib.plots(rmse,alphas,r2score,train,test)

