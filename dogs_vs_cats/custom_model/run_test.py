import random
import sys
import psutil
import numpy as np
import os 
import tensorflow as tf
import estimator
import pandas as pd

def run_test(covnet_estimator,
                 sess,
                 experiment_dir,
                 dataset_path = "../persistent_storage/test_images_1000.npy"):
    
    
    
    X_te = np.load(dataset_path)
    
    

    
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    
    total_t = sess.run(tf.contrib.framework.get_global_step())
    total_t_copy = total_t


    metrics = []

    #calculate accuracy for train set 
    preds_tr = covnet_estimator.predict(sess,X_te) #probabilities        
    y_pred = sess.run(tf.reduce_max(preds_tr, axis=1))

    df = pd.DataFrame({'id':np.arange(X_te.shape[0]) , 'label':y_pred})
    
    return df
    



tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

experiment_dir = os.path.abspath("./experiment/") 
covnet_estimator = estimator.Estimator(scope="conv_net",summaries_dir=experiment_dir,name="cnn_2")


import time
start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    df = run_test(covnet_estimator,sess,experiment_dir=experiment_dir)
    print(df.head())
    df.to_csv("submission.csv", index=False)

end = time.time()
