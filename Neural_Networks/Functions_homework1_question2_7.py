#######OMML Assignment ####################
import numpy as np
import OMML_gen_data as data
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


def mse_train(y_train,y,vec,roh):

    mse = np.mean((y_train.reshape(y_train.shape[0],1) - y.reshape(y.shape[0],1))**2) 
    reg_term = roh*np.sqrt(np.sum(abs(vec)**2))
    return mse + reg_term

def mse_test(y_test,y):

    mse = np.mean((y_test.reshape(y_test.shape[0],1) - y.reshape(y.shape[0],1))**2)
    return mse


####################### Question 2 part 1 ############################################

def init_parameters_mlp2(Num_nuerons = 10, X = data.X_train):

    No_nueron_units = Num_nuerons
    #np.random.seed(1772576)
    v = np.random.rand(No_nueron_units,1)                 #vector of length equal to no of neurons N x 1
    w = np.random.rand(No_nueron_units,X.shape[1])        #matrix of dimension N x n
    b = np.random.rand(No_nueron_units,1)                 # N x 1
    #shapes = (v.shape,w.shape,b.shape)
    #vctr = toVector(v,w,b)
    
    return(v,w,b)



def optimize_mlp(X_train,y_train,X_test,y_test,param_dct):
    X = X_train
    yy = y_train.reshape(len(y_train),1)
            
    #param_dct is the dictionary of parameters of RBF network in Q1 Part 2  
    
    
    #test set predictions on these parameters
    v,w_hatmlp,b_hatmlp = init_parameters_mlp2(param_dct['N'],X_train)
    res = minimize(cost_function_mlp, v,args=(w_hatmlp,b_hatmlp,X_train,y_train,param_dct['roh'],param_dct['sigma']),method='Nelder-Mead',options={'disp': True,'maxiter':30000})
    v_hatmlp = res.x
    y_hat_mlp = fx(X_test,v_hatmlp,w_hatmlp,b_hatmlp,param_dct['sigma'])
    test_se = cost_function_mlp(v_hatmlp,w_hatmlp,b_hatmlp,X_test,y_test,param_dct['roh'],param_dct['sigma'])
    
    return y_hat_mlp,v_hatmlp,w_hatmlp,b_hatmlp,param_dct,test_se


def gridsearch_mlp(X_train,y_train,X_test,y_test,N_vec=[4,10,20],roh_vec=[0.0001,0.01,0.05,0.1,3,0.0003],sigma_vec = [0.001,0.1,0.05]):
    X = X_train
    yy = y_train.reshape(len(y_train),1)

    dumpdct={}
    cnt = 0
    for N in N_vec:
        v,w_hatmlp,b_hatmlp = init_parameters_mlp2(N,X)
        for roh in roh_vec:
            for sigma in sigma_vec:
                cnt = cnt+1
                res = minimize(cost_function_mlp, v,args=(w_hatmlp,b_hatmlp,X_train,y_train,roh,sigma),method='BFGS',options={'disp': True,'maxiter':30000})
                v_hatmlp = res.x
                #res.fun                                                                   #train error
                err = cost_function_mlp(v_hatmlp,w_hatmlp,b_hatmlp,X_test,y_test,roh,sigma)    #test error
                dumpdct[cnt] = {'N':N,'roh':roh,'sigma':sigma,'train_error':res.fun,'test_error':err}

                
                
    #select parameters with lowest test and train errors
    param_dct = select_parameters_mlp(dumpdct)
    
    #test set predictions on these parameters
    v,w_hatmlp,b_hatmlp = init_parameters_mlp2(param_dct['N'],X_train)

    res = minimize(cost_function_mlp, v,args=(w_hatmlp,b_hatmlp,X_train,y_train,param_dct['roh'],param_dct['sigma']),method='BFGS',options={'disp': True,'maxiter':30000})
    v_hatmlp = res.x
    y_hat_mlp = fx(X_test,v_hatmlp,w_hatmlp,b_hatmlp,param_dct['sigma'])   #predictions on test set
    test_se = cost_function_mlp(v_hatmlp,w_hatmlp,b_hatmlp,X_test,y_test,param_dct['roh'],param_dct['sigma'])
    
    return y_hat_mlp,v_hatmlp,w_hatmlp,b_hatmlp,param_dct,test_se


def cost_function_mlp(v,w,b,X,y,roh,sigma):
    
#     v                  #vector of length equal to no of neurons N x 1
#     w                  #matrix of dimension N x n
#     b                  # N x 1

    

    
    reg_term = roh*np.sqrt(np.sum(abs(w)**2) + np.sum(abs(b)**2) + np.sum(abs(v)**2))
    cost = np.sum((fx(X,v,w,b,sigma).reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term
    
    return cost

def select_parameters_mlp(dct_object):
    # Find parameters having lowest test and train error.
    tst_error_list = []
    train_error_list = []
    for i in dct_object:
        tst_error_list.append(dct_object[i]['test_error'])
        train_error_list.append(dct_object[i]['train_error'])

    a,b = tst_error_list.index(min(tst_error_list)),train_error_list.index(min(train_error_list))
    return dct_object[a+1]

def g(t,sigma):
    val = (1-np.exp(-sigma*t))/(1+np.exp(-sigma*t))
    return val


def fx(X,v,w,b,sig):
    
    #dimensions of X  = p x n
    

    #v                  #vector of length equal to no of neurons N x 1
    #w                  #matrix of dimension N x n
    #b                  # N x 1
    
    
    
    f_xp = np.dot(v.T,g((np.dot(w,X.T) - b),sig))           #1xN x Nxp = 1xp vector
    return f_xp.T






####################### Question 2 part 2 ############################################
# Estimate centroids for RBF network by K-Means
from sklearn.cluster import KMeans

# initialize parameters for RBF 
def init_parameters_rbf_2(Num_nuerons = 10, X = data.X_train):

    No_nueron_units = Num_nuerons
    np.random.seed(1772576)
    
    
    # estimate cluster centroids by K Means
    est = KMeans(n_clusters=No_nueron_units)
    est_obj = est.fit(X)
    cluster_centroids = est_obj.cluster_centers_
    
    v = np.random.rand(No_nueron_units,1)                 #vector of length equal to no of neurons N x 1
    
    return(v,cluster_centroids)


# returns optimal values for weights v
def optimize(X_train,y_train,X_test,y_test,param_dct):
    X = X_train
    yy = y_train.reshape(len(y_train),1)
            
    #param_dct is the dictionary of parameters of RBF network in Q1 Part 2  
    
    
    #test set predictions on these parameters
    v,c_hatrbf = init_parameters_rbf_2(param_dct['N'],X_train)
    res = minimize(cost_function_rbf_2, v,args=(c_hatrbf,X_train,y_train,param_dct['roh'],param_dct['sigma']),method='BFGS',options={'disp':True,'maxiter':30000})
    v_hatrbf = res.x
    
    return v_hatrbf,c_hatrbf,res

#measures the loss
def cost_function_rbf_2(v,c,X,y,roh,sigma):
    
#     v               #vector of length equal to no of neurons N x 1
#     w               #matrix of dimension N x n
#     b               # N x 1

    
    reg_term = roh*np.sqrt(np.sum(abs(v)**2) + np.sum(abs(c)**2))
    cost = np.sum((fx_rbf(X,v,c,sigma).reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term
    
    return cost


# pack vector
def toVector_rbf(v, c):
    assert v.shape == (v.shape[0], v.shape[1])
    assert c.shape == (c.shape[0], c.shape[1])
    return np.hstack([v.flatten(), c.flatten()])

# unpack vector
def toWZ_rbf(vec,shapes):
    
    vshape,cshape = shapes
    v1 = vec[:vshape[0]*vshape[1]].reshape(vshape[0],vshape[1])
    v2 = vec[-cshape[0]*cshape[1]:].reshape(cshape[0],cshape[1])
    return v1,v2

#gauss function
def gauss(t,ci,sigma):
    
    #c=c.reshape(len(c),1)
    k = np.sqrt(np.sum((abs(t-ci)**2),axis = 1))
    val = np.exp(-((k/sigma)**2))
    
    return val


#output function
def fx_rbf(X,v,c,sigma):
    
    #dimensions of X  = p x n
    #v  vector of length equal to no of neurons N x 1
    #c  vector of length equal to no of neurons N x 1
    if v.shape[0]!=c.shape[0]:
        raise(Exception('v,c, should have same length !'))
    
    v = v.reshape(len(v),1)
    gauss_mat = np.empty([X.shape[0],v.shape[0]])
    
    for i in range(c.shape[0]):
        gauss_mat[:,i] = gauss(X,c[i],sigma)
        
    
    
    out = np.dot(gauss_mat,v)   #pxN x Nx1 = px1 vector
                                                    
    return out




