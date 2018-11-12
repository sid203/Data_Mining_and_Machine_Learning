import random
import sys
import psutil
import numpy as np
import os 
import tensorflow as tf

def make_batches(batch , number_of_batches , X_tr, y_tr , X_te, y_te):
    train_batch_size = int(X_tr.shape[0] / number_of_batches)
    test_batch_size = int(X_te.shape[0]/number_of_batches)
    
    start_ind = batch * train_batch_size 
    end_index = start_ind + train_batch_size
    
    x_train = X_tr[start_ind:end_index,:,:,:]
    y_train = y_tr[start_ind:end_index,:]

    start_ind = batch * test_batch_size 
    end_index = start_ind + test_batch_size

    x_test = X_te[start_ind:end_index,:,:,:]
    y_test = y_te[start_ind:end_index,:]
    return x_train,y_train,x_test,y_test


def run_training(covnet_estimator,
                 sess,
                 experiment_dir,
                 dataset_path = "../persistent_storage/train_images_1000.npy",
                 label_path = "../persistent_storage/train_labels_1000.npy",
                 max_epochs = 100):
    
    
    
    data = np.load(dataset_path)
    labels = np.load(label_path)
    
    #split for train and test 
    X_tr = data[:int(0.6*data.shape[0]),:,:,:]
    y_tr = labels[:int(0.6*labels.shape[0]),:]
    X_te = data[int(0.6*data.shape[0]):,:,:,:]
    y_te = labels[int(0.6*data.shape[0]):,:]
    
    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()
    
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

    
    for epoch in range(max_epochs):
        
        saver.save(tf.get_default_session(), checkpoint_path)
        
        metrics = []
        number_of_batches = 10
        for batch in range(number_of_batches):
            x_train,y_train,x_test,y_test = make_batches(batch,number_of_batches,X_tr,y_tr,X_te,y_te)
            #update estimator
            loss = covnet_estimator.update(sess,x_train,y_train)
            #calculate accuracy for train set 
            preds_tr = covnet_estimator.predict(sess,x_train)
            y_pred_tr = tf.argmax(preds_tr, dimension=1)
            y_true_tr = tf.argmax(y_train, dimension=1) 
            correct_prediction = tf.equal(y_pred_tr, y_true_tr)
            acc_tr = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) 
            
            
            
            
            #test set predictions and accuracy
            loss_test = covnet_estimator.test_set_loss(sess,x_test,y_test)[0]
            preds = covnet_estimator.predict(sess,x_test)
            y_pred_cls = tf.argmax(preds, dimension=1)
            y_true_cls = tf.argmax(y_test, dimension=1)
            #calculate accuracy for test set 
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            acc_te = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) 
            
            

            
            
            
            print("Epoch {} , batch_number {}/10 , train_loss {} ,test_loss {}, test_accuracy {} , train accuracy {}".format(epoch,batch,loss,loss_test,acc_te,acc_tr),end='\n')
            metrics.append([loss,loss_test,acc_tr,acc_te])
            
            # add hidden weights to summary file
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv_net'):
                if 'fully_connected' and 'Conv' in var.name:
                    covnet_estimator.log_histogram(tag= var.name.split(':')[0],step=total_t,values=sess.run(var))
    
            timestep_summary = tf.Summary()
            timestep_summary.value.add(simple_value=acc_tr, tag="timestep/train_accuracy")
            timestep_summary.value.add(simple_value=acc_te, tag="timestep/test_accuracy")
            timestep_summary.value.add(simple_value=loss, tag="timestep/train_loss")
            timestep_summary.value.add(simple_value=loss_test, tag="timestep/test_loss")

            covnet_estimator.summary_writer.add_summary(timestep_summary, total_t)
            
    
        total_t +=1
        
        
        # Add summaries to tensorboard
        avg_metrics = [sum(e)/len(metrics) for e in zip(*metrics)] 


        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=avg_metrics[0], tag="episode/average_train_loss")
        episode_summary.value.add(simple_value=avg_metrics[1], tag="episode/average_test_loss")
        episode_summary.value.add(simple_value=avg_metrics[2], tag="episode/average_train_accuracy")
        episode_summary.value.add(simple_value=avg_metrics[3], tag="episode/average_test_accuracy")
        episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
        episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
        covnet_estimator.summary_writer.add_summary(episode_summary, (total_t_copy)+epoch)
        covnet_estimator.summary_writer.add_graph(sess.graph)
        covnet_estimator.summary_writer.flush()
        
        
        
            
