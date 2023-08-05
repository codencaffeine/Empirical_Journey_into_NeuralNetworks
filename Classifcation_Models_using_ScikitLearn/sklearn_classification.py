if __name__ == "__main__":
    #Importing the necessary libraries
    import os
    import pandas as pd

    #* Saving the path of the dataset in a variable
    train_path = "./train.csv"
    test_path = "./test.csv"
    #-----------------------------------------------------------
    #* Load the csv file of training data using pandas into a dataframe and store it in the variable train_data
    train_data = pd.read_csv(train_path)
    #* Load the csv file of test data using pandas into a dataframe and store it in the variable test_data
    test_data = pd.read_csv(test_path)
    #* Check how the dataframe looks like using the .head() method of pandas. This will output first 5 rows of the DF(dataframe)
    # train_data.head(5)
    #* This will cout the number of women in the dataset. .value_counts() method is used to see the number of entries per unique value in the column
    #* Since we have men and women as the two uniqe values, [1] will give us the number of women and not men
    train_data["Sex"].value_counts()[1]
    #* Use this functon to see the overall information of the DF includinfg the datatype, number of entries, missing values etc.
    # train_data.info()
    #* A very poerful function to use if you want to see statistical properties like mean, median, count, etc of the DF
    # train_data.describe()
    #-----------------------------------------------------------
    """
    The BaseEstimator class is the base class for all scikit-learn estimators (models). Estimators in scikit-learn are objects that learn from 
    data using the fit() method and can make predictions on new data using the predict() method.
    The TransformerMixin class is another base class provided by scikit-learn.
    It is intended for custom transformer classes that implement data transformations, such as preprocessing steps in machine learning pipelines.
    The TransformerMixin class provides a fit_transform() method, which combines the fit() and transform() methods. It allows for easy creation 
    of custom transformers that can learn from data during fitting and apply the transformation to the data in a single call.
    By using these base classes, you can create custom machine learning models and transformers that integrate seamlessly into scikit-learn
    pipelines and can be used in conjunction with other scikit-learn components. These classes ensure a consistent interface, making it 
    easier to combine different parts of a machine learning workflow and enabling a more modular and flexible design of machine learning systems.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    #* Create a custom class named DataFrameSelector
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        #*  This is the constructor of the DataFrameSelector class. It takes a parameter attribute_names, which is a list of column names to be selected 
        #*  from the input DataFrame
        def __init__(self, attribute_names):
            #* The constructor stores the input attribute_names in the attribute_names instance variable of the class
            self.attribute_names = attribute_names
        #* The fit method is required for any transformer class. However, since the DataFrameSelector class does not need to learn anything from the data, 
        #* this method does nothing and simply returns self
        def fit(self, X, y=None):
            return self
        #* The transform method is where the actual transformation is performed. It takes the input data X, which is a DataFrame, and returns the selected columns 
        #* specified in attribute_names
        def transform(self, X):
            return X[self.attribute_names]
    #-----------------------------------------------------------
    """
    We will now import the Pipeline Class from the pipeline module of sklearn
    This will helps us, as the name suggests, club up several sequential data processing or modelling steps into one
    sequence. We will have a custom transformer and an imputer in our pipeline as shown below
    """
    from sklearn.pipeline import Pipeline
    """
    In real-world datasets, it's common to encounter missing values in various features or columns. 
    Missing data can cause issues during the training of machine learning models, as many algorithms 
    cannot handle missing values directly. The SimpleImputer class provides a convenient way to fill 
    in missing values with a specified strategy.
    """
    from sklearn.impute import SimpleImputer
    #* We create an instance of the Pipeline class called num_pipeline
    num_pipeline = Pipeline([
            #* The object takes in arguments as the name of the step (data transformer) which is "select_numeric"
            #* The second argument is the custom data transformer DataFrameSelector which takes in numeric attributes of our DF
            ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
            #* The second step in the pipeline is the "imputer" (SimpleImputer), and the strategy selected is median
            ("imputer", SimpleImputer(strategy="median")),
        ])
    #* Fit the pipeline on the training data using the method .fit_transform()
    num_pipeline.fit_transform(train_data)
    #-----------------------------------------------------------
    """
    We will now create a custom Imputer using th base classes BaseEstimator and TransformerMixin
    The class will have two methods in it: .fit() and .transform()
    """
    class MostFrequentImputer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            #* Here, it calculates the most frequent value for each column in the input DataFrame X and stores it in the 
            #* most_frequent_ attribute as a pandas Series. The Series contains the most frequent value for each column, and the index of the 
            #* Series is set to the column names of X.
            self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                            index=X.columns)
            return self
            #* The transformation is done by the transform method of this class, by calling .fillna() on input X with the most frequent value 
            #* (self.most_frequent_) as the parameter. Any missing values in X will be replaced with the corresponding most frequent value from that attribute
        def transform(self, X, y=None):
            return X.fillna(self.most_frequent_)
    #-----------------------------------------------------------
    #* Import the OneHotEncoder class from the preprcessing module of sklearn
    from sklearn.preprocessing import OneHotEncoder
    #* Now we create a pipeline named cat_pipeline for preprocessing categorical features in the Titanic survivor dataset
    cat_pipeline = Pipeline([
            #* The first step in the pipeline is the custom transformer DataFrameSelector. It is named "select_cat" and selects the columns "Pclass", "Sex", and "Embarked" 
            #* from the input DataFrame (train_data).
            ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
            #* The second step in the pipeline is the custom transformer MostFrequentImputer(), which imputes missing values in the selected categorical columns using the 
            #* most frequent value for each column.
            ("imputer", MostFrequentImputer()),
            #* The third step in the pipeline is scikit-learn's OneHotEncoder, which is used to perform one-hot encoding on the selected categorical columns
            ("cat_encoder", OneHotEncoder(sparse=False)),
        ])
    #* Fit the pipeline on the trai_data using .fit_transform() method 
    cat_pipeline.fit_transform(train_data)
    #-----------------------------------------------------------
    """
    We import FeatureUnion from scikit-learn's pipeline module, which allows us to combine multiple pipelines into a single pipeline.
    Since we have two peipelines num_pipeline and cat_pipeline, for numerical data and categorical data rspectively, we need to now combine both of them
    into one pipeline. 
    """
    from sklearn.pipeline import FeatureUnion
    #* create an instance of the FeatureUnion class 
    preprocess_pipeline = FeatureUnion(transformer_list=[
        #* The transformer list parameter will have the two custom data transformers we created, and we name them as strings
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])
    #* Now we call the fit_transform method on the pipeline object and pass the train_data to it
    X_train = preprocess_pipeline.fit_transform(train_data)
    #* We store the attribute that we want to redict in the y_train variable
    y_train = train_data["Survived"]
    #-----------------------------------------------------------
    """
    The SVC stands for Support Vector Classifiers
    We will be using SVM (Suppor Vector Machine) to classify between survivors and non-survivors
    """
    from sklearn.svm import SVC
    #* We create an instance of the class SVC (Create an SVM) and initialize it with the gamma="auto" parameter, which determines 
    #* the kernel coefficient for the "rbf" (radial basis function) kernel. We set the random state to 40 for result reproducibility
    svm_clf = SVC(gamma="auto", random_state=40)
    #* Now we fit the SVM classifier on the train_data and the corresponding target values
    svm_clf.fit(X_train, y_train)
    #* Now we do transformations on the test data too, so that we can make predictions using it
    X_test = preprocess_pipeline.transform(test_data)
    #* The .predict() method is called on the classifier, passing the transformed test data
    y_pred = svm_clf.predict(X_test)
    #* To validate the predictions we got, let us use the cross_val_score function of the model_selection module of sklearn
    from sklearn.model_selection import cross_val_score
    #* We pass the model name, training data, the labels of the training data, and cv (number of folds for the validation)
    svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
    #* We now compute the mean of the scores from the SVM classifier
    svm_scores.mean()
    #* To use another classifier based on Random Forest method (Collection of decision trees), we import the RandomForestClassifier class 
    #* of the ensemble module of sklearn
    from sklearn.ensemble import RandomForestClassifier
    #* We create an instance of the class (a model) by passing in the parameters n_estimator and random_state
    #* n_estimator is the number of decision trees in the forest and random_state is set to a number 40 for reproducibility
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=40)
    #* To cross validate, we use cross_val_score again
    forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
    #* Taking mean of the scores
    forest_scores.mean()