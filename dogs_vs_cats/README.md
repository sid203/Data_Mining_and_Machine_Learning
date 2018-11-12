## Classying Cats and Dogs using InceptionV3 and Custom Model. 

### Description 
This Project provides a starter code to classify images by training a pretrained model InceptionV3. Also a custom architecture would be used which is loosely inspired from VGG-16. 
The dataset is available on kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data). 


### Steps to execute scripts 

1. Unzip test images under your_directory/train/ folder
2. Unzip train images under your_directory/train/ folder
3. Unzip contents of zip file under your_directory
4. Run preprocessing.py : 
	1. cd to /your_directory
	2. execute python preprocessing.py


	This script will create a folder "persistent_storage" - your_directory/persistent_storage
	Under persistent_storage there will be 3 files : train_images_500.npy , test_images_500.npy , train_labels_500.npy
	These files are of the dimensions 1000 x 224 x 224 x 3 , 500 x 224 x 224 x 3 , 1000 x 224 x 224 x 3 respectively. 
	train_images_500.npy : 1000 samples from train
	train_labels_500.npy : 1000 labels of train files
	test_images_500.npy : 1000 samples from train
	The IMAGE_HEIGHT , IMAGE_WIDTH can be changed by editing the preprocessing.py file

5. To run custom model , migrate to folder your_directory/custom_model/ : 
	1. cd to this directory and execture on terminal : python run_training.py 
	This will create ./experiments/ folder. 
	/experiments folder has checkpoints and summaries_conv_net folders which contain model checkpoints and tensorboard summary file respectively. 

	REMARK : run_training.py assumes preprocessed images to be 64 x 64 x 3 because tensorflow can't handle bigger sizes (unless you are parallelizing input data). 
	So run preprocessing.py again and change IMAGE_HEIGHT = 64, IMAGE_WIDTH = 64 train_name = "/train_images_1000.npy" train_labels = "/train_labels_1000.npy" test_name = "/test_images_1000.npy"
	It will create files train_images_1000.npy , train_labels_1000.npy, test_images_1000.npy under your_directory/persistent_storage

	2. Once the training is done, you can view the tensorboard summary files by executing this command : tensorboard --logdir your_directory/custom_model/experiment/summaries_conv_net/ 
	3. Now you can run run_test.py to predict on test images ( which are assumed to be preprocessed , by default the test file is test_images_1000.npy ). This will create submission.csv at the end. 
	4. The model I used was a scaled down version of VGG-16 with different stride params under the name "cnn_2"
	4. All other scripts are intermediate scripts used in the backend. 

6. To run pre-trained model, migrte to your_directory/pretrained_model. 
	1. execute the script train_script.py and it will write inception_model.h5 file under your_directory/pretrained_model/
	2. Now you can run test_script.py and it will write submission.csv 
	REMARK : This model assumed images to be 224 x 224 x 3 because the model used is InceptionV3 and these were the default image dimensions. So the files used will be
	 ../persistent_storage/train_images_500.npy , "../persistent_storage/train_labels_500.npy" , "../persistent_storage/test_images_500.npy"
