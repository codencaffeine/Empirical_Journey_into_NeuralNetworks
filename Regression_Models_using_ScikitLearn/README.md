## You can find the code in Regression_Models_using_ScikitLearn/sklearn_regression.py and the contents are as follows:
---
1. We use the California Housing Dataset to make predictions on the current housing prices based on the 1990 census.
2. We import the pandas library to load the csv data into a dataframe
3. We use the StratifiedShuffleSplit Class of the model_selection module of sklearn to split the data into test and train sets(just for demonstration purpose)
4. We create an imputer, used to fill in the missing values in our dataset. We make use of the SimpleImputer Class of the impute module of sklearn
5. We see how to use One Hot Encoding to encode the categorical data
6. We use BaseEstimator and TransformerMixin as the base classes to create our custom classes for estimators and data transformations respectively
7. We then use the StandarScaler Class from the preprocessing module to standardize the dataset
8. Pipeline Class is then used to create a pipeline or a sequence of data transformations on numerical data
9. To create the full pipeline for both numerical and categorical data, we make use of ColumnTransformer class
10. After the data processing part, we create several models like Linear regression model, Decision tree model, Random Forest model, etc.
11. We also demonstrate how to implement the cross_val_score to validate our predictions.