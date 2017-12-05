#######OMML Assignment ####################
import numpy as np
import OMML_gen_data as data
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


# returns mse + regularization term
def mse_train(y_train,y,vec,roh):

    mse = np.mean((y_train.reshape(y_train.shape[0],1) - y.reshape(y.shape[0],1))**2) 
    reg_term = roh*np.sqrt(np.sum(abs(vec)**2))
    return mse + reg_term

# returns test + regularization term
def mse_test(y_test,y):

    mse = np.mean((y_test.reshape(y_test.shape[0],1) - y.reshape(y.shape[0],1))**2)
    return mse


##### Question 1 part 1 ##########

# initializes the values of the network
def init_parameters(Num_nuerons = 10, X = data.X_train):

    No_nueron_units = Num_nuerons
    np.random.seed(1772576)
    v = np.random.rand(No_nueron_units,1)                 #vector of length equal to no of neurons N x 1
    w = np.random.rand(No_nueron_units,X.shape[1])        #matrix of dimension N x n
    b = np.random.rand(No_nueron_units,1)                 # N x 1
    shapes = (v.shape,w.shape,b.shape)
    vctr = toVector(v,w,b)                                # pack the parameters in a vector  
    
    return(vctr,shapes)

# packs the parameters into a vector
def toVector(v, w, b):
    assert v.shape == (v.shape[0], v.shape[1])
    assert w.shape == (w.shape[0], w.shape[1])
    assert b.shape == (b.shape[0], b.shape[1])
    return np.hstack([v.flatten(), w.flatten(),b.flatten()])

# unpacks the vector back into parameters
def toWZ(vec,shapes):
    
    vshape,wshape,bshape = shapes
    v1 = vec[:vshape[0]*vshape[1]].reshape(vshape[0],vshape[1])
    v2 = vec[vshape[0]*vshape[1]:vshape[0]*vshape[1]+wshape[0]*wshape[1]].reshape(wshape[0],wshape[1])
    v3 = vec[-bshape[0]*bshape[1]:].reshape(bshape[0],bshape[1])
    return v1,v2,v3

# activation function
def g(t,sigma):
    val = (1-np.exp(-sigma*t))/(1+np.exp(-sigma*t))
    return val

# output function
def fx(X,v,w,b,sig):
    
    #dimensions of X  = p x n
    
    #(v,w,b) = params
    #v                  #vector of length equal to no of neurons N x 1
    #w                  #matrix of dimension N x n
    #b                  # N x 1
    
    
    
    f_xp = np.dot(v.T,g((np.dot(w,X.T) - b),sig))           #1xN x Nxp = 1xp vector
    return f_xp.T


# measures the loss 
def cost_function(vctr,X,y,roh,shapes,sigma):
    

#     v                 #vector of length equal to no of neurons N x 1
#     w                 #matrix of dimension N x n
#     b                 # N x 1
    
    v,w,b = toWZ(vctr,shapes)   # unpack the vector
    
    reg_term = roh*np.sqrt(np.sum(abs(w)**2) + np.sum(abs(b)**2) + np.sum(abs(v)**2))                       # regularization term
    cost = np.sum((fx(X,v,w,b,sigma).reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term # loss + regularization
    
    return cost



# returns a dict with optimal hyperparameters

def select_parameters_mlp(dct_object):
    # Find parameters having lowest test and train error.
    tst_error_list = []
    train_error_list = []
    for i in dct_object:
        tst_error_list.append(dct_object[i]['test_error'])
        train_error_list.append(dct_object[i]['train_error'])

    a,b = tst_error_list.index(min(tst_error_list)),train_error_list.index(min(train_error_list))
    return dct_object[a+1]


    

# perform a gridsearch and return prediction on optimal hyperparameters
def gridsearch(X_train,y_train,X_test,y_test,N_vec=[4,10,20],roh_vec=[0.0001,0.01,0.05,0.1,3,0.0003],sigma_vec = [0.001,0.1,0.05]):
    X = X_train
    yy = y_train.reshape(len(y_train),1)

    dumpdct={}
    cnt = 0
    for N in N_vec:
        vctr,shapes = init_parameters(N,X)   #init parameters
        for roh in roh_vec:
            for sigma in sigma_vec:
                cnt = cnt+1
                res = minimize(cost_function, vctr,args=(X,yy,roh,shapes,sigma),method='BFGS',options={'disp': True,'maxiter':30000})
                v_hat,w_hat,b_hat = toWZ(res.x,shapes)
                
                err = cost_function(res.x,X_test,y_test,roh,shapes,sigma)    #test error
                dumpdct[cnt] = {'N':N,'roh':roh,'sigma':sigma,'train_error':res.fun,'test_error':err}  # res.fun is train error

                
                
    #select parameters with lowest test and train errors
    param_dct = select_parameters_mlp(dumpdct)
    
    #test set predictions on these parameters
    vctr,shapes = init_parameters(param_dct['N'],X_train)

    res_test = minimize(cost_function, vctr,args=(X,yy,param_dct['roh'],shapes,param_dct['sigma']),method='BFGS',options={'disp': True,'maxiter':30000})
    v_hat,w_hat,b_hat = toWZ(res_test.x,shapes)
    y_hat = fx(X_test,v_hat,w_hat,b_hat,param_dct['sigma'])
    
    return y_hat,v_hat,w_hat,b_hat,param_dct


##### Question 1 part 2 ##########

# packs the parameters into a vector
def toVector_rbf(v, c):
    assert v.shape == (v.shape[0], v.shape[1])
    assert c.shape == (c.shape[0], c.shape[1])
    return np.hstack([v.flatten(), c.flatten()])

# unpacks the vector back into parameters
def toWZ_rbf(vec,shapes):
    
    vshape,cshape = shapes
    v1 = vec[:vshape[0]*vshape[1]].reshape(vshape[0],vshape[1])
    v2 = vec[-cshape[0]*cshape[1]:].reshape(cshape[0],cshape[1])
    return v1,v2

# activation function
def gauss(t,ci,sigma):
    
    #c=c.reshape(len(c),1)
    k = np.sqrt(np.sum((abs(t-ci)**2),axis = 1))
    val = np.exp(-((k/sigma)**2))
    
    return val

# output function
def fx_rbf(X,v,c,sigma):
    
    #dimensions of X  = p x n
    #v  vector of length equal to no of neurons N x 1
    #c  vector of length equal to no of neurons N x 1
    if v.shape[0]!=c.shape[0]:
        raise(Exception('v,c, should have same length !'))        #raise error if v and c are not equal in length
    
    v = v.reshape(len(v),1)
    gauss_mat = np.empty([X.shape[0],v.shape[0]])                 # dimensions are p x N        
    
    for i in range(c.shape[0]):
        gauss_mat[:,i] = gauss(X,c[i],sigma)                      #matrix holding values of activation function for every training example and center c
        
    
    
    out = np.dot(gauss_mat,v)   #pxN x Nx1 = px1 vector
                                                    
    return out


# measures the loss 
def cost_function_rbf(vctr,X,y,roh,shapes,sigma):
    

#     v                                                 #vector of length equal to no of neurons N x 1
#     w                                                 #matrix of dimension N x n
#     b                                                 # N x 1

    
    v,c = toWZ_rbf(vctr,shapes)
    
    reg_term = roh*np.sqrt(np.sum(abs(v)**2) + np.sum(abs(c)**2))
    cost = np.sum((fx_rbf(X,v,c,sigma).reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term
    
    return cost



# initialize parameters for RBF 
def init_parameters_rbf(Num_nuerons = 10, X = data.X_train):

    No_nueron_units = Num_nuerons
    #np.random.seed(1772576)
    v = np.random.rand(No_nueron_units,1)                 #vector of length equal to no of neurons N x 1
    c = np.random.rand(No_nueron_units,X.shape[1])        # N x n 
    shapes = (v.shape,c.shape)
    vctr = toVector_rbf(v,c)
    
    return(vctr,shapes)
    

def select_parameters_rbf(dct_object):
    # Find parameters having lowest test and train error.
    tst_error_list = []
    train_error_list = []
    for i in dct_object:
        tst_error_list.append(dct_object[i]['test_error'])
        train_error_list.append(dct_object[i]['train_error'])

    a,b = tst_error_list.index(min(tst_error_list)),train_error_list.index(min(train_error_list))
    return dct_object[a+1]


#perform gridsearch and predict values at optimal hyperparameters
def gridsearch_rbf(X_train,y_train,X_test,y_test,N_vec=[4,10,20],roh_vec=[0.0001,0.01,0.05,0.1,3,0.0003],sigma_vec = [0.001,0.1,0.05]):
    X = X_train
    yy = y_train.reshape(len(y_train),1)

    dumpdct={}
    cnt = 0
    for N in N_vec:
        vctr,shapes = init_parameters_rbf(N,X)
        for roh in roh_vec:
            for sigma in sigma_vec:
                cnt = cnt+1
                res = minimize(cost_function_rbf, vctr,args=(X,yy,roh,shapes,sigma),method='Nelder-Mead',options={'disp': True,'maxiter':30000})
                v_hat,c_hat = toWZ_rbf(res.x,shapes)
                
                err = cost_function_rbf(res.x,X_test,y_test,roh,shapes,sigma)    #test error
                dumpdct[cnt] = {'N':N,'roh':roh,'sigma':sigma,'train_error':res.fun,'test_error':err}   #res.fun is train error
            
    #select parameters with lowest test and train errors
    param_dct = select_parameters_rbf(dumpdct)
    
    #test set predictions on these parameters
    vctr,shapes = init_parameters_rbf(param_dct['N'],X_train)

    res_test = minimize(cost_function_rbf, vctr,args=(X,yy,param_dct['roh'],shapes,param_dct['sigma']),method='Nelder-Mead',options={'disp': True,'maxiter':30000})
    v_hatrbf,c_hatrbf = toWZ_rbf(res_test.x,shapes)
    y_hat_rbf = fx_rbf(X_test,v_hatrbf,c_hatrbf,param_dct['sigma'])
    
    return y_hat_rbf,v_hatrbf,c_hatrbf,param_dct


