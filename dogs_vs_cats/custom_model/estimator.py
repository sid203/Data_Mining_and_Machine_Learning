import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import models


#_R_MEAN = 123.68
#_G_MEAN = 116.78
#_B_MEAN = 103.94

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CHANNELS = 3

#TRAIN_DIR = '../train/'
#TEST_DIR = '../test/'

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """




    
    def __init__(self, state_shape = [None,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS] ,convs = [(32,5,2),(64,5,2),(128,5,2)], hiddens = [1024], num_classes = 2, name = "cnn" ,layer_norm = False,scope="estimator", summaries_dir=None):

     
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        self.hiddens = hiddens
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model( convs,num_classes,hiddens,layer_norm,state_shape = state_shape,name = name)
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)



    def _build_model(self,convs,num_classes,hiddens,layer_norm,state_shape, name = "cnn"):

        self.X_pl = tf.placeholder(shape=state_shape, dtype=tf.float16, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None,2], dtype=tf.float16, name="y")
        # Integer id of which action was selected
        #self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl)
        #batch_size = tf.shape(self.X_pl)[0]

        # Three mlp layers
        mod = models.select_model(name)
        out = mod(X,convs,hiddens,num_classes,layer_norm) #FC Layer output
        
        
        self.predictions = tf.nn.softmax(out) #probabilities 

        # Calculate the loss
        self.losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_pl, logits = out)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

#         correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("probabilities", self.predictions),
            tf.summary.scalar("max_prob", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts probabilities.

        Args:
          sess: Tensorflow session
          s: State input of shape [None, 64,64,3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 64,64,3]
          y: Targets of shape [batch_size,2]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y}
        summaries,global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(),self.train_op, self.loss],    # tf.contrib.framework.get_global_step()
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
    
    def test_set_loss(self,sess,s,y):
        feed_dict = { self.X_pl: s, self.y_pl: y}
        loss = sess.run([self.loss], feed_dict) #run without optimizer
        return loss

    
    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()
