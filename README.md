## Brushing over the concept of `overfitting` which was demonstrated in our first simple implementation of a Neural network trying to solve a classification problem based on MNIST Dataset.

## What is overfitting in the simplest of terms?
### **It's the scenario when your model memorizes your training data and loses the ability to perform well on any other data!**

1. For example, let's compare `YOU` with a `model`. Suppose you are to give an exam on basic algebra. You decided to learn and memorize the textbook word to word, as a strategy to score the highest. So, by doing this, you know the answers to all the questions mentioned in your textbook. Wonderfull!
2. Let's say one of the questions you memorized was: $(x + y)^{2}$ = $x^{2}$ +  $y^{2}$ + 2xy.
3. In the examination, you are asked to give the formula to solve the following: $(a + b)^{2}$ = ?
4. You never saw this exact example in the textbook! And since you just blindly memorized every line in it, you couldn't understand the meaning behind this particular equation, and you ended up failing the test. This is a highly undesirable outcome.
5. The same analogy works for our model as well. Overfitting makes the model learn the training data too well, but the model will perform terribly on new data (example: test data or validation data.)
6. To avoid this, we only let our model capture the essense of our training data. The moment it starts to overfit, we stop the training. OR! there are methods that will restrict the model and stop it from overfitting.
7. Let's see some of the methods used often.

## The methods used to avoid or reduce overfitting will depend on the dataset you are usign and the algorithm you are implementing. Thus, you need to make different approaches on the same problem and find the best method that works for your model.

1. **`Splitting the data into train, test and validation sets:`** This is a given basic step that you must follow. This step allows you to monitor the performance of your model on the test data and detect overfitting.

2. **`Regularization techniques:`** This is a method of adding a penalty term to the loss function and discourage a model from becoming too complex, or having large values of `weights`! Thus by doing this, we are forcing a model to take only small values which makes the weight distribution more regular. The following are some of the mothods:

  `A)` **L1 regularization:** In this method, the cost added to the loss function is proportional to the absolute value of the co-efficients of the weights. Don't worry, we will have an implementation of this soon.

  `B)` **L2 regularization(weight decay method):** In this method, the cost added to the loss function is propotional to the sqaure of the weight co-efficients. 

3. **`Early stopping:`** In this method, we track the performance on the validation set during the training process. When we see a sharp increase in the validation error, it's a sure indication of overfitting. This is the indication to stop the training!

4. **`Adding Dropout:`** Dropout is a method applied to different layers in a Neural Network to randomly set a fraction of the output features to 0. For example, if a particular layer is supposed to output a vector `[ 0.5 2.1 1.6 0.8 ]`, and if you apply dropout to that layer so that it drops ramdomly 2 feature outputs, you might get something of this sort as your ouput vector: `[ 0.5 0 1.6 0]`. 

5. **`Pruning:`** This method is used for networks that comprises of decision trees. It in simple terms, cuts down those branches of the decision trees that have not much influence on the final predictions.

6. **`Ensembling:`** This method has techniques like `bagging` and `boosting` (the later being one of the most popular nowadays, example: XGBoost) to avoid overfitting in algorithms like random forests and so on, where the predictions are an average of the results from different models. 

7. **`Increasing Training data:`** Overfitting is almost a certain event when your model is too complex(consisting of many layers or parameters) and your DataSet is too sparse(few samples). In other words, you need a large amount of data is you are going to use highly parameterised and heavy models like Bert and so on. If you try to solve a problem using Bert and your Dataset only consists of 500 samples, you are sure to have an overfitting problem. To solve this, acquire more data.

8. **`Data Augmentation:`** You will see this technique most often in computer vision, where you randomly flip, rotate, crop or add noise to certain data samples(images) and add these transformed samples back into the training data. This method increases the diversity of your data samples and might help you develop a more robust model.

9. **`Dimensionality reduction:`** This technique is not the best of solutions available, but still can be used as an effective tool for certain kind of problems. When the data is too dense(high dimensional data), the model that is used to train on such data tends to be very complex. This will also invite the problem of overfitting. PCA(Principal Component Analysis) is one of the dimensionality reduction techniques that can help you get rid of some dimensions of the data that don't meaningfully contribute to the outcome.
