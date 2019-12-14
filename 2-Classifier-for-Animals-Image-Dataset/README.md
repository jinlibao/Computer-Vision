# Project 2: Classifier for Animals Image Dataset

## Description

This project is to build upon the code given in Chapter 12 to develop a good CNN classifier for the Animals dataset. The Animals dataset is a easier toy problem by today’s standards, as it is smaller (only 3000 images) and has only three cases. It is thus a nice dataset to play with and learn the various techniques for designing CNNs without each design iteration taking too much time.

First, design a CNN which does the best job overall of classifying the validation set. So that all results can be easily compared, use the same validation set that Chapter 12 uses for its result (i.e., set the random state to 42). Along the way, you should try all of the methods we have discussed (changing the number of layers, type of layers, neurons per layer, activation function, learning rate, momentum, batch normalization, data augmentation and regularization). Document how you tried changing all these items - it should be a systematic, well thought-out study, not just randomly trying different values. What factors seemed to most influence the overall performance? Did you observe overfitting or underfitting? Show those results. Remember to measure performance using accuracy, precision, recall, and F1 score.

Once you’ve found your best CNN, try find a reduced version of it that requires less computations and/or memory to implementations, clear documentation, etc. Show me that you understand how to design a CNN. A formal written report documenting the above is expected and will be graded for grammar and clearness of exposition.

## Data

* [Animals Image Dataset (Dogs, Cats, and Panda) on Kaggle](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)
