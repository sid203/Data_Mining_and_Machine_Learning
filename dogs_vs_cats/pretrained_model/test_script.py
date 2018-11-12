import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3


model = load_model('./inception_model.h5')
y_te = np.load("../persistent_storage/test_images_500.npy")



def test_image_preds(y_te,model):
    fig=plt.figure()

    for num,data in enumerate(y_te):
        # cat: [1,0]
        # dog: [0,1]
        img_num = num
        img_data = data
        y = fig.add_subplot(3,5,num+1)
        orig = img_data
        data = img_data.reshape(-1,IMG_HEIGHT,IMG_WIDTH,CHANNELS)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1: str_label='Dog'
        else: str_label='Cat'

        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()



def get_dataframe(y_te,model,df_name = "/submission.csv"):
    
    prob = []
    img_list = []
    for num,data in enumerate(y_te):
            img_num = num
            img_data = data
            data = img_data.reshape(-1,IMG_HEIGHT,IMG_WIDTH,CHANNELS)
            model_out = model.predict([data])[0]
            img_list.append(img_num)
            prob.append(model_out[1])
            
    df = pd.DataFrame({'id':img_list , 'label':prob})
    print(df.head())
    df.to_csv(os.getcwd()+df_name, index=False)
    return df




preds = get_dataframe(y_te,model)
#display some images
n_samples = 15
test_image_preds(y_te[:n_samples,:,:,:],model)
