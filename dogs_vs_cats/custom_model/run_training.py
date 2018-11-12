import time
import estimator
import train
import tensorflow as tf
import os


tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

experiment_dir = os.path.abspath("./experiment/") 
covnet_estimator = estimator.Estimator(scope="conv_net",summaries_dir=experiment_dir,name="cnn_2")



start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train.run_training(covnet_estimator,sess,experiment_dir=experiment_dir,max_epochs=100,dataset_path = "../persistent_storage/train_images_1000.npy",label_path = "../persistent_storage/train_labels_1000.npy")

end = time.time()
