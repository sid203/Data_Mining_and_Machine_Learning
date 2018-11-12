import os, cv2, random
import numpy as np 
import pandas as pd 


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3

#specify train and test directories here
TRAIN_DIR = './train/'
TEST_DIR = './test/'
train_name = "/train_images_500.npy"
train_labels = "/train_labels_500.npy"
test_name = "/test_images_500.npy"

def get_filenames(num_samples = 500):
    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

    # using only 1000 samples 
    train_images = train_dogs[:num_samples] + train_cats[:num_samples]
    random.shuffle(train_images)
    test_images =  test_images[:num_samples]
    
    return train_images,test_images


def resize_image(file_path):        
    # tf functions are slow no idea why so ill use opencv
    #image_decoded = tf.image.decode_jpeg(image_file, channels=3)    
    #resize_fn = tf.image.resize_image_with_crop_or_pad
    #image_resized = resize_fn(image_decoded, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
    
    return img#,np.asarray(image_resized.eval()),tf.to_float(image_resized)


def normalize_image(image,means=[_R_MEAN,_G_MEAN,_B_MEAN]): #borrowed from VGG preprocessing

    
    for i in range(CHANNELS):
        image[:,:,i] = image[:,:,i] - means[i]    
        
    return image
  


def get_label(img_path):
    # [1 0] for cat [0 1] for dog
    if 'dog' in img_path:
        return np.array([0,1])
    if 'cat' in img_path:
        return np.array([1,0]) 
    
    
    
def prepare_data(images,train_flag=True):    
    count = len(images)
    data = np.ndarray((count, IMAGE_HEIGHT, IMAGE_WIDTH,CHANNELS), dtype=np.uint8)
    labels = np.ndarray((count,2),dtype = np.uint8) #binary classification
    
    for i, image_file in enumerate(images):
        image = resize_image(image_file)
        image = normalize_image(image)
        data[i] = image
        if train_flag:
            labels[i] = get_label(image_file)
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    return data,labels



#prepare datasets
tr_image_path,te_image_path = get_filenames()
tr_image , tr_labels = prepare_data(tr_image_path)
te_image , _ = prepare_data(te_image_path,train_flag=False)

print("Train shape: {}".format(tr_image.shape))
print("Test shape: {}".format(te_image.shape))

#write datasets
save_dir = os.path.abspath("./persistent_storage/") 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


np.save(save_dir+train_name, tr_image)
del tr_image #to free memory
np.save(save_dir+train_labels, tr_labels)
del tr_labels
np.save(save_dir+test_name, te_image)
del te_image


