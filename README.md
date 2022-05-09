# Flower-Classification

Creating a data input pipeline from TensorFlow flower classifier tutorial

https://www.tensorflow.org/hub/tutorials/image_feature_vector

We will use a technique called transfer learning where we take a pre-trained network (trained on about a million general images), use it to extract features, and train a new layer on top for our own task of classifying images of flowers.

# flower_classifier.py

Juptyer notebook detailing model training pipeline.

The flowers dataset consists of images of flowers with 5 possible class labels.

When training a machine learning model, we split our data into training and test datasets. We will train the model on our training data and then evaluate how well the model performs on data it has never seen - the test set.

# test_model.py

Unit testing

# application.py

Flask app to connect model to front end.

![image](https://user-images.githubusercontent.com/64989388/167458804-06b110e8-af5f-4d23-a24f-3ffbbd5eaf11.png)
