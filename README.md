# Plant Disease Classification using Convolutional Neural Networks

This repository contains the code and dataset used to train a Convolutional Neural Network (CNN) for the classification of plant diseases. The model is trained on the "Augmented Plant Diseases" dataset, which contains images of various plants affected by different diseases.

## Dataset

The "Augmented Plant Diseases" dataset is divided into two subsets: training and validation. The dataset can be obtained from [link to the dataset source if available]. The images in the dataset have been resized to (256x256) pixels.

## Model Architecture

The CNN model is implemented using Keras, a popular deep learning library. The architecture consists of multiple convolutional layers followed by max-pooling layers to downsample the spatial dimensions of the features. The final fully connected layers are used for classification. The model's architecture is as follows:

1. Input: (256, 256, 3) - RGB images with a size of 256x256 pixels.
2. Conv2D (32 filters, 3x3 kernel, ReLU activation, padding='same').
3. Conv2D (32 filters, 3x3 kernel, ReLU activation, padding='same').
4. MaxPooling2D (3x3 pool size).
5. Conv2D (64 filters, 3x3 kernel, ReLU activation, padding='same').
6. Conv2D (64 filters, 3x3 kernel, ReLU activation, padding='same').
7. MaxPooling2D (3x3 pool size).
8. Conv2D (128 filters, 3x3 kernel, ReLU activation, padding='same').
9. Conv2D (128 filters, 3x3 kernel, ReLU activation, padding='same').
10. MaxPooling2D (3x3 pool size).
11. Conv2D (256 filters, 3x3 kernel, ReLU activation, padding='same').
12. Conv2D (256 filters, 3x3 kernel, ReLU activation, padding='same').
13. Conv2D (512 filters, 5x5 kernel, ReLU activation, padding='same').
14. Conv2D (512 filters, 5x5 kernel, ReLU activation, padding='same').
15. Flatten.
16. Dense (1568 units, ReLU activation).
17. Dropout (rate=0.5).
18. Dense (38 units, softmax activation) - Output layer for 38 disease classes.

## Data Preprocessing

Before feeding the images to the model, they are rescaled to values between 0 and 1 by dividing each pixel value by 255. This step is essential for better convergence during training.

## Training

The model is trained using the Adam optimizer with a learning rate of 0.0001. The loss function used is sparse categorical cross-entropy, as this is a multi-class classification problem. The model is trained for 10 epochs on the training dataset, and the performance is evaluated on the validation dataset after each epoch.

## Evaluation

After training the model, it is evaluated on the validation dataset to measure its performance. The evaluation includes calculating the loss and accuracy of the model on the validation set.

## Results

The final evaluation of the model provides insights into its performance on the unseen validation data. The accuracy metric represents the percentage of correctly classified instances, while the loss metric indicates how well the model's predictions align with the ground-truth labels.

The true labels and model predictions are stored for further analysis and comparison.

## Note

Please ensure that you have the required dependencies and the "Augmented Plant Diseases" dataset in the specified directory before running the code. The dataset can be obtained from [provide the dataset source link if available].

Feel free to experiment with the model architecture, hyperparameters, and data augmentation techniques to improve the model's performance further.

If you have any questions or suggestions, feel free to open an issue or pull request.

Happy coding!
