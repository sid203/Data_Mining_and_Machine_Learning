import Functions_homework1_question2_7 as fun


X_train = fun.data.X_train
X_test = fun.data.X_test
y_train = fun.data.y_train
y_test = fun.data.y_test

print('\n')
print('###### Question 2 part 1 ###################')
###### Question 2 part 1 ###################

## Uncomment to run the gridsearch ###
#import time

#start_mlp = time.time()
#y_hat_mlp, v_hatmlp, w_hatmlp, b_hatmlp, dct_mlp2,test_error_mlp = gridsearch_mlp(X_train,y_train,X_test,y_test)
#end_mlp = time.time()
#print(end_mlp-start_mlp)



# Dict with optimal parameters #
dct_mlp={'N': 20,
 'roh': 0.0001,
 'sigma': 0.1,
 'test_error': 0.017926432537633191,
 'train_error': 0.015139482893578783}


v,w_hatmlp,b_hatmlp = fun.init_parameters_mlp2(dct_mlp['N'],X_train)

import time
start_1 = time.time()

res = fun.minimize(fun.cost_function_mlp, v,args=(w_hatmlp,b_hatmlp,X_train,y_train,dct_mlp['roh'],dct_mlp['sigma']),method='BFGS',options={'disp': True,'maxiter':30000})
end_1 = time.time()
v_hatmlp = res.x




#Test set prediction#
y_hat_mlp = fun.fx(X_test,v_hatmlp,w_hatmlp,b_hatmlp,dct_mlp['sigma'])
mse_te_mlp = fun.mse_test(y_test,y_hat_mlp)

#Train set prediction#
y_train_hat = fun.fx(X_train,v_hatmlp,w_hatmlp,b_hatmlp,dct_mlp['sigma'])
mse_tr_mlp = fun.mse_train(y_train,y_train_hat,res.x,dct_mlp['roh'])

dct_output = {'MSE train_set + regularization':mse_tr_mlp,
'MSE test_set':mse_te_mlp,
'Computational time in seconds':end_1-start_1,
'Function Evaluations':res.nfev,
'Gradient Evaluations':res.njev }

print('\n')
print('Output Dict for Question2 part 1 ')
print('\n')
print(dct_output)

####### Print the approximating function ###################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_hat_mlp.flatten(), cmap=cm.jet, 
                       linewidth=0.2, antialiased=True,alpha=0.5)

fig.colorbar(surf, shrink=10, aspect=3)

ax.set_xlabel('xtest_1')
ax.set_ylabel('xtest_2')
ax.set_zlabel('test_predictions_y_hat')
plt.title(' Plot for Approximating function found ')
plt.show()


fig2 = plt.figure()
ax2 = Axes3D(fig2)
surf = ax2.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_test.flatten(), cmap=cm.jet, linewidth=0.1,alpha=0.5)
fig2.colorbar(surf, shrink=10, aspect=3)

ax2.set_xlabel('xtest_1')
ax2.set_ylabel('xtest_2')
ax2.set_zlabel('y_test')
plt.title(' Plot for Original function ')
plt.show()



print('\n')
print('###### Question 2 part 2 ###################')
###### Question 2 part 2 ###################



dct_rbf = {'N': 20,
 'roh': 0.0001,
 'sigma': 0.1,
 'test_error': 0.034291308898091712,
 'train_error': 0.0007897255290297369}


import time

start_rbf2 = time.time()
v_hatrbf,c_hatrbf,result = fun.optimize(X_train,y_train,X_test,y_test,dct_rbf)
end_rbf2 = time.time()
print(end_rbf2-start_rbf2)


#Test set predictions
y_hat_rbf_te = fun.fx_rbf(X_test,v_hatrbf,c_hatrbf,dct_rbf['sigma'])
mse_te_rbf = fun.mse_test(y_test,y_hat_rbf_te)


#Train set prediction#
y_hat_rbf_tr = fun.fx_rbf(X_train,v_hatrbf,c_hatrbf, dct_rbf['sigma'])
mse_tr_rbf = fun.mse_train(y_train,y_hat_rbf_tr,result.x,dct_rbf['roh'])

dct_output_rbf = {'MSE train_set + regularization':mse_tr_rbf,
'MSE test_set':mse_te_rbf,
'Computational time in seconds':end_rbf2-start_rbf2,
'Function Evaluations':result.nfev,
'Gradient Evaluations':result.njev }

print('\n')
print('Output Dict for Question2 part 2 ')
print('\n')
print(dct_output_rbf)

####### Print the approximating function ###################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_hat_rbf_te.flatten(), cmap=cm.jet, 
                       linewidth=0.2, antialiased=True,alpha=0.5)

fig.colorbar(surf, shrink=10, aspect=3)

ax.set_xlabel('xtest_1')
ax.set_ylabel('xtest_2')
ax.set_zlabel('test_predictions_y_hat')
plt.title(' Plot for Approximating function found ')
plt.show()


fig2 = plt.figure()
ax2 = Axes3D(fig2)
surf = ax2.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_test.flatten(), cmap=cm.jet, linewidth=0.1,alpha=0.5)
fig2.colorbar(surf, shrink=10, aspect=3)

ax2.set_xlabel('xtest_1')
ax2.set_ylabel('xtest_2')
ax2.set_zlabel('y_test')
plt.title(' Plot for Original function ')
plt.show()
