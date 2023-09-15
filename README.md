Neural-Networks
-------
In the example, I use Google Colab, TensorFlow, KerasTuner, scikit-learn libraries to create a binary classifier that can predict whether funding applicants will be successful. It uses a CSV containing more than 34,000 organizations that have received and the status of their application. 

Highlights: 
-------
* Using Pandas and scikit-learn’s StandardScaler(), I preprocess the dataset.
* Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
* Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.


Compile, Train, and Evaluate the Model:
-------
* Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
* Create the first hidden layer and choose an appropriate activation function.
* Create an output layer with an appropriate activation function.
* Compile and train the model.
* Create a callback that saves the model's weights every five epochs.
* Evaluate the model using the test data to determine the loss and accuracy.

Summary:
-------
I initialized a Keras Sequential model for evaluating the dataset, and began experimenting with a variety of configurations manually. I expected that our model would preform best with initial rectilinear, softmax or linear activation functions and an sigmoid output layer. Employing a sigmoid function as your output layer yeild improved accuracy when the target variable is binary. Given the type of data, it did not seem approprite to employ other activations better suited to non-linear data. These assumptions seem to be merited given the results below. After various iterations, and manually changing these values, I was acheiving a prediction of approximately 73%. After including 'NAME' in the analysis, I was able to manually acheive an accuracy of 75%. However, I was keen to see the accuracy improved, so I initialized a keras tuner to evaluate a wide range of neurons, layers, and activation functions. After 13 hours of processing this is the current best model with an accuracy of 80.03%:

activation: 'linear'
first_units: 15
num_layers: 2
units_0: 11
units_1: 97
units_2: 57
units_3: 15
units_4: 87
tuner/epochs: 6
tuner/initial_epoch: 2
tuner/bracket: 3
tuner/round: 1
tuner/trial_id: '0383'

Due to some impatience on my part, I interrupted the model tuning after 13 hours and cooked an egg on my motherboard. Additional time spent allowing the tuner to fully complete may yet yeild a model with better performance. It is my recommendation that the first tuner with 'max_epochs=50' and 'hyperband_iterations=10' be initialized again and run to completion.

The limitations of this approach is that the tuner seems not to blend different types of activations, or did not run long enough to offer combinations of activations for our layers. Additional manual manipluation of the 'best-model' may also provide increased performance beyond 80% accuracy. For example, all the layers in our 'best-model' are linear, so tweaking them, or adding additional softmax or relu layers may increase performance. Manual tweaking has yet to achieve results greater than 80%.

An attempt was made to find a better model within the parameters suggested by the first tuner, but has yet to yeild better results. The best results thus far for this second tuner has been 79.95%, acheiving results quite close to the performance of the initial tuner.

Lastly, a combined approach using an initial PCA to determine the most important components, may well lead to improvement in computational time. By determining the most impactful principal components, the dataset and features could be reduced. It is my recommendation that a PCA analysis be completed prior to preprocessing, giving a measure of the most important features to be used in a subsequent binary classifier. However, initial attempts at manual manipulation of the dataset by reducing features has not yeilded any improvment in model accuracy.
