#######OMML Assignment ####################
import numpy as np
import OMML_gen_data as data
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# train mse
def mse_train(y_train,y,vec,roh):

    mse = np.mean((y_train.reshape(y_train.shape[0],1) - y.reshape(y.shape[0],1))**2) 
    reg_term = roh*np.sqrt(np.sum(abs(vec)**2))
    return mse + reg_term

#test mse
def mse_test(y_test,y):

    mse = np.mean((y_test.reshape(y_test.shape[0],1) - y.reshape(y.shape[0],1))**2)
    return mse


################ Question 3 ########################

### Two Block Decomposition Method ####



# activation function
def gauss_3(t,ci,sigma):
    
    #c=c.reshape(len(c),1)
    k = np.sqrt(np.sum((abs(t-ci)**2),axis = 1))
    k2 = abs(t - ci)                              # required for calculating the gradient
    val = np.exp(-((k/sigma)**2))
    
    return val,k2

    
# output function
def fx_rbf_3(X,v,c,sigma):
    
    #dimensions of X  = p x n
    #v  vector of length equal to no of neurons N x 1
    #c  vector of length equal to no of neurons N x 1
    c = c.reshape(v.shape[0],X.shape[1])
    if v.shape[0]!=c.shape[0]:
        raise(Exception('v,c, should have same length !'))
    
    v = v.reshape(len(v),1)
    gauss_mat = np.empty([X.shape[0],v.shape[0]])
    l1_norm_mat = []            # this will hold l1_norm between every center and training example
    
    for i in range(c.shape[0]):
        g1,g2 = gauss_3(X,c[i],sigma)
        gauss_mat[:,i] = g1
        l1_norm_mat.append(g2)
    
    
    out = np.dot(gauss_mat,v)   #pxN x Nx1 = px1 vector
                            
    return out,gauss_mat,l1_norm_mat


# measures the loss
def cost_function_rbf(v,c,X,y,roh,sigma):
    

#     v                  #vector of length equal to no of neurons N x 1
#     w                  #matrix of dimension N x n
#     b                  # N x 1

    
    
    
    reg_term = roh*np.sqrt(np.sum(abs(v)**2) + np.sum(abs(c)**2))    #regularization term
    cost = np.sum((fx_rbf_3(X,v,c,sigma)[0].reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term  #loss + reg_term



    return cost

# another wrapper function with the same functionality but required in two_block() with a different name
def cost_function_rbf_c(c,v,X,y,roh,sigma):
    
#     v                  #vector of length equal to no of neurons N x 1
#     w                  #matrix of dimension N x n
#     b                  # N x 1            # N x 1

    
    
    reg_term = roh*np.sqrt(np.sum(abs(v)**2) + np.sum(abs(c)**2))   #regularization term
    cost = np.sum((fx_rbf_3(X,v,c,sigma)[0].reshape(X.shape[0],1) - y.reshape(len(y),1))**2)/(2*len(y)) + reg_term   #loss + reg_term



    return cost


# calculates the gradient for v
def grad_v(v,c,X,y,roh,sigma):
    
    val,gauss_matrix,dump = fx_rbf_3(X,v,c,sigma)
    values = []
    values = [np.dot(((val.reshape(X.shape[0],1) - y.reshape(len(y),1)).T),gauss_matrix[:,i].reshape(X.shape[0],1)) + (abs(2*v[i])*roh) for i in range(v.shape[0])]
    val1 = (np.array(values) / X.shape[0]) 
    return val1.reshape(val1.shape[0])   #vector of gradients for dE/dv1 , dE/dv2 .....

# calculates the gradient for centers c
def grad_c(c,v,X,y,roh,sigma):
    
    c = c.reshape(v.shape[0],X.shape[1])
    val,gauss_matrix,l1_mat = fx_rbf_3(X,v,c,sigma)
    vect = [np.dot((val.reshape(X.shape[0],1) - y.reshape(len(y),1)).T ,(gauss_matrix[:,i] * v[i] * l1_mat[i][0:,0].reshape(X.shape[0]))) + (abs(2*c[i][0])*roh) for i in range(c.shape[0])]
    vect1 = [np.dot((val.reshape(X.shape[0],1) - y.reshape(len(y),1)).T ,(gauss_matrix[:,i] * v[i] * l1_mat[i][0:,1].reshape(X.shape[0]))) + (abs(2*c[i][1])*roh) for i in range(c.shape[0])] 
    value = np.hstack([vect,vect1]).reshape(len(vect)*2,1)
    value = (2*value)/(X.shape[0]*(sigma**2))
    
    return value.reshape(value.shape[0])  #vector of gradients for dE/dc1 , dE/dc2 .....
    


import numdifftools as nd      #required for calculating jacobian 

# Estimate centroids for RBF network by K-Means
from sklearn.cluster import KMeans

# initialize parameters
def init_parameters_rbf_2(Num_nuerons = 10, X = data.X_train):

    No_nueron_units = Num_nuerons
    np.random.seed(1772576)
    
    
    # estimate cluster centroids by K Means
    est = KMeans(n_clusters=No_nueron_units)
    est_obj = est.fit(X)
    cluster_centroids = est_obj.cluster_centers_
    
    v = np.random.rand(No_nueron_units,1)                 #vector of length equal to no of neurons N x 1

    
    return(v,cluster_centroids)

# performs two block
def two_block(X,y,dct):
    
    
    J = nd.Jacobian(cost_function_rbf_c)      # to calculate the gradient at particular values 
    v,c = init_parameters_rbf_2(dct['N'],X)   # init parameters using K means for centroids c

    tol = [1e-6,1e-3]
    lst = []
    func_eval = 0                             # to hold total function evaluations
    grad_eval = 0                             # to hold total gradient evaluations 
    counter = 0                               # to hold iterations
    while(True):
        ### optimize v ####
        res_v = minimize(cost_function_rbf, v,args=(c,X,y,dct['roh'],dct['sigma'])     #minimize by passing gradients for v 
                 ,method='CG',jac=grad_v,options={'disp': True,'maxiter':30000})
        v_hat = res_v.x
        func_eval =func_eval+ res_v.nfev          #number of function evaluations

        norm = np.linalg.norm(J(c.flatten(),v_hat,X,y,dct['roh'],dct['sigma']), ord=2)    # l2norm of gradient dE/dc evaluated at v_hat 
     
        if (norm >= tol[0] and norm <= tol[1]):    #exit condition
            break
        
        ### optimize c ###

        res_c = minimize(cost_function_rbf_c, c.flatten(),args=(v_hat,X,y,dct['roh'],dct['sigma'])
                 ,method='CG',options={'disp': True,'maxiter':30000})

        grad_eval = grad_eval+res_c.njev          #number of gradient evaluations

        #lst.append(cost_function_rbf(v_hat,c,X,y,dct['roh'],dct['sigma']))    # hold the loss at every iteration
        ### update parameters #####
        v = v_hat
        c = res_c.x
        counter = counter + 1
    print('total iter',counter)
    return v_hat,c,func_eval,grad_eval

