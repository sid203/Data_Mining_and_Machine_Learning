import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt



def combine_data(train,test):
#This function returns combination of train and test data in variable house_data
#The SalesPrice column which is the true set of labels is returned in true_data
    temp=train
    house_data=pd.concat([temp.drop(['SalePrice'],axis=1),test],axis=0) 
    true_data=train['SalePrice']
    del temp
    return house_data,true_data



def checknans(df):
#Function to check the number of nans in every column

    nan_cols=df.columns[df.isnull().any()].tolist()
    return(df[nan_cols].isnull().sum())




def cat_variables_counts(col,df):
#Function returns count of repeating records (or levels) in a column
    return(df[col].value_counts())



def cat_fillvalue(col,df,val):
#This function fills the nan values in a column by supplied value
    df.loc[df[col].isnull(),col] = val




def genpolynomials(x,a):
#Generates values equal to x^a
    return(x**a)




#Ridge regression ( or regularization)
def ridge_analysis(alphas,Train_data,Norm_data_true,true_data):
#Returns r2 score and rmse for different values of alpha
    score=[]
    rmse=[]
    
    for i in alphas:
        rr=Ridge(i,fit_intercept=True)
        rr.fit(Train_data.values, Norm_data_true.values)
        score.append(rr.score(Train_data.values, Norm_data_true.values)) # R^2 score
        rpred=np.exp(rr.predict(Train_data))
        rmse.append(np.sqrt(np.mean(np.square((rpred-true_data.values)))))

    return score,rmse


def plots(rmse,alphas,score,train,test):
#Creates and saves different plots

    NaN_data,dump=combine_data(train,test)
    res=checknans(NaN_data)
    res.sort_values(inplace=False)
    res.plot.bar()
    plt.title('Plot of NaN values')
    plt.savefig('NaN_plot.png')						#Plot showing different NaN values
    plt.gcf().clear()

    rmse=pd.Series(rmse,index=alphas)
    rmse.plot(title = "Rmse_Curve")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.title('RMSE_Curve')
    plt.savefig('Rmse_Curve.png')					#RMSE_Curve
    plt.gcf().clear()

    corr = train.corr()
    corr.sort_values(["SalePrice"], ascending = False, inplace = True)
    corr_plot=pd.DataFrame(corr.SalePrice)
    corr_plot.plot.bar()
    plt.title('Correlation of columns with SalesPrice')
    plt.savefig('Correlations.png')					#Plot showing Correlation of columns
    plt.gcf().clear()

    results = pd.Series(score,index=alphas)
    results.columns=["R Square Score"] 
    results.plot(kind="bar",title="Alpha Scores")
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    plt.title('Alpha Scores')
    plt.savefig('Alpha_Scores.png')					#Plotting Score with alphas
    plt.gcf().clear()





