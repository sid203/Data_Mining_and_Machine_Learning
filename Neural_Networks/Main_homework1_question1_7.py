import Functions_homework1_question1_7 as fun


X_train = fun.data.X_train
X_test = fun.data.X_test
y_train = fun.data.y_train
y_test = fun.data.y_test

print('\n')
print('###### Question 1 part 1 ###################')
###### Question 1 part 1 ###################

## Uncomment to run the gridsearch ###
#import time

#start = time.time()
#y_hat,v_hat,w_hat,b_hat,dct_mlp = fun.gridsearch(X_train,y_train,X_test,y_test)
#end = time.time()
#print(end - start)



# Dict with optimal parameters #
dct_mlp={'N': 10,
 'roh': 0.0001,
 'sigma': 0.1,
 'train_error': 0.009188975569789974}

#test set predictions on these parameters
vctr,shapes = fun.init_parameters(dct_mlp['N'],X_train)

import time
start_1 = time.time()

res_tr = fun.minimize(fun.cost_function, vctr,args=(X_train,y_train,dct_mlp['roh'],shapes,dct_mlp['sigma']),method='BFGS',options={'disp': True,'maxiter':30000})
v_hat,w_hat,b_hat = fun.toWZ(res_tr.x,shapes)

end_1 = time.time()

#Test set prediction#
y_hat = fun.fx(X_test,v_hat,w_hat,b_hat,dct_mlp['sigma'])
mse_te_mlp = fun.mse_test(y_test,y_hat)

#Train set prediction#
y_train_hat = fun.fx(X_train,v_hat,w_hat,b_hat,dct_mlp['sigma'])
mse_tr_mlp = fun.mse_train(y_train,y_train_hat,res_tr.x,dct_mlp['roh'])

dct_output = {'MSE train_set + regularization':mse_tr_mlp,
'MSE test_set':mse_te_mlp,
'Computational time in seconds':end_1-start_1,
'Function Evaluations':res_tr.nfev,
'Gradient Evaluations':res_tr.njev }

print('Output Dict for Question1 part 1 ')
print('\n')
print(dct_output)

####### Print the approximating function ###################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_hat.flatten(), cmap=cm.jet, 
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







##### Uncomment to view to predictions #########################

#print(pd.DataFrame({'x':y_hat.reshape(len(y_hat)),'y':y_test}))


print('\n')
print('###### Question 1 part 2 ###################')
###### Question 1 part 2 ###################

import time

#start_rbf = time.time()
#y_hat_rbf,v_hatrbf,c_hatrbf,dct_rbf = gridsearch_rbf(X_train,y_train,X_test,y_test)
#end_rbf = time.time()
#print(end_rbf - start_rbf)



# Dict with optimal parameters #
dct_rbf={'N': 20,
 'roh': 0.0001,
 'sigma': 0.1,
 'test_error': 0.029392371360539207,
 'train_error': 0.0046727970828470515}

#test set predictions on these parameters
vctr,shapes = fun.init_parameters_rbf(dct_rbf['N'],X_train)

import time
start_2 = time.time()

res_tr_2 = fun.minimize(fun.cost_function_rbf, vctr,args=(X_train,y_train,dct_rbf['roh'],shapes,dct_rbf['sigma']),method='Nelder-Mead',options={'disp': True,'maxiter':30000})
v_hatrbf,c_hatrbf = fun.toWZ_rbf(res_tr_2.x,shapes)


end_2 = time.time()

#Test set prediction#
y_hat_rbf = fun.fx_rbf(X_test,v_hatrbf,c_hatrbf,dct_rbf['sigma'])
mse_te_rbf = fun.mse_test(y_test,y_hat_rbf)

#Train set prediction#
y_train_hat_rbf = fun.fx_rbf(X_train,v_hatrbf,c_hatrbf,dct_rbf['sigma'])
mse_tr_rbf = fun.mse_train(y_train,y_train_hat_rbf,res_tr_2.x,dct_mlp['roh'])


dct_output_rbf = {'MSE train_set + regularization':mse_tr_rbf,
'MSE test_set':mse_te_rbf,
'Computational time in seconds':end_2-start_2,
'Function Evaluations':res_tr_2.nfev,
'Gradient Evaluations':'Gradient not used in Nelder-Mead method' }

print('Output Dict for Question1 part 2 ')
print('\n')
print(dct_output_rbf)


####### Print the approximating function ###################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_hat_rbf.flatten(), cmap=cm.jet, 
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


##### Uncomment to view to predictions #########################

#print(pd.DataFrame({'x':y_hat_rbf.reshape(len(y_hat_rbf)),'y':y_test}))
