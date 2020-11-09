## Self-Driving-Car
# Dataset 
https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view
# Objective
The objective of this project is to predict the steering angle when the car moves forward . The angle is predicted frame by frame .
# Tools Used
Used Keras with tensorflow backend to train the model as well as to pre-process the image in Google colab . Then downloaded the weights . The downloaded weights were used to predict the steering angle using OpenCV in the local computer .
# Model Architecture
The model was trained on a modified version of Nvidia Net where there were Conv Layers of increasing number of filters and then a couple of dense layers. Before training, the upper half portion of the image was removed as it didn't play a vital role in determining the steering angle .
