import Functions_homework1_question3_7 as fun



X_train = fun.data.X_train
X_test = fun.data.X_test
y_train = fun.data.y_train
y_test = fun.data.y_test

print('\n')
print('###### Two Block ############')
###### Two block ###################

print('Estimated time: 250 seconds')


# Dict with optimal parameters #
dct_rbf = {'N': 20,
 'roh': 0.0001,
 'sigma': 0.1,
 'test_error': 0.034291308898091712,
 'train_error': 0.0007897255290297369}

import time
start_1 = time.time()

v_hat_bl,c_hat_bl,f_eval,grad_eval = fun.two_block(X_train,y_train,dct_rbf)
end_1 = time.time()



#Test set prediction#
y_hat_bl = fun.fx_rbf_3(X_test,v_hat_bl,c_hat_bl,dct_rbf['sigma'])[0] 
mse_te_bl = fun.mse_test(y_test,y_hat_bl)





#Train set prediction#
y_hat_bl_tr = fun.fx_rbf_3(X_train,v_hat_bl,c_hat_bl,dct_rbf['sigma'])[0] 
mse_tr_bl = fun.mse_train(y_train,y_hat_bl_tr,fun.np.hstack([v_hat_bl.flatten(), c_hat_bl.flatten()]),dct_rbf['roh'])

dct_output = {'MSE train_set + regularization':mse_tr_bl,
'MSE test_set':mse_te_bl,
'Computational time in seconds':end_1-start_1,
'Function Evaluations':f_eval,
'Gradient Evaluations':grad_eval }

print('\n')
print('Output Dict for Question3 ')
print('\n')
print(dct_output)


####### Print the approximating function ###################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X_test[:,0].flatten(), X_test[:,1].flatten(), y_hat_bl.flatten(), cmap=cm.jet, 
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

