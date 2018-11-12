import tensorflow as tf
import tensorflow.contrib.layers as layers



def cnn_to_mlp(X, convs, hiddens , num_classes, layer_norm , reuse=False ):
    
    out = tf.to_float(X)
    for num_outputs, kernel_size, stride in convs:
        out = layers.convolution2d(out,
                                   num_outputs=num_outputs,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   activation_fn=tf.nn.relu)
        out = layers.max_pool2d(inputs=out,kernel_size=(2,2))

    #flatten
    out = layers.flatten(out)

    for hidden in hiddens:
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
        if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
        out = tf.nn.relu(out)

    labels = layers.fully_connected(out, num_outputs=num_classes, activation_fn=None) 
    #labels = tf.nn.softmax(out) 
    return labels



def cnn_2(X, convs, hiddens , num_classes, layer_norm , reuse=False ):
    
    out = tf.to_float(X)

    #print(out)
    out = layers.convolution2d(out,num_outputs=32,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    #print(out)
    out = layers.convolution2d(out,num_outputs=32,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    #print(out)
    out = layers.max_pool2d(inputs=out,kernel_size=(2,2))
    #print(out)
    
    out = layers.convolution2d(out,num_outputs=64,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.convolution2d(out,num_outputs=64,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.max_pool2d(inputs=out,kernel_size=(2,2))

    out = layers.convolution2d(out,num_outputs=128,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.convolution2d(out,num_outputs=128,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.max_pool2d(inputs=out,kernel_size=(2,2))    
    
    out = layers.convolution2d(out,num_outputs=256,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.convolution2d(out,num_outputs=256,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
    out = layers.max_pool2d(inputs=out,kernel_size=(2,2))
    #print(out)
    out = layers.flatten(out)
    out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
    out = layers.dropout(out,0.5)
    out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
    out = layers.dropout(out,0.5)
    
    labels = layers.fully_connected(out, num_outputs=num_classes, activation_fn=None) 
    #labels = tf.nn.softmax(out) 
    return labels    
    

def select_model(name = "cnn"):


    """Initializes neural network model,summary_writer (for tensorboard).

    Parameters
    ----------
    name : string
        type of neural network model for estimation. 
        avail models : "mlp","lstm"

    Returns
    -------
    model : function 
        returns the selected model as a function. 
    """

    
    if name == "cnn":
        return cnn_to_mlp

    if name == "cnn_2":
        return cnn_2




