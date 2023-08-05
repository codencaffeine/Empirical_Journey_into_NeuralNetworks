if __name__ == "__main__":
    #Importing the necessary libraries
    import numpy as np
    import sklearn
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    #------------------------------------------------------------------------------------------------------------------

    #Loading the csv data using pandas and saving it as a data frame
    path = "./housing.csv"
    housing = pd.read_csv(path)
    #*.info() prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.
    # print(housing.info())
    #* The resulting object of .value_counts() will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.
    # ans = housing["ocean_proximity"].value_counts()
    # print(ans)
    #* .describe() generates descriptive statistics like mean, standard deviation, count, etc of the series data or the data frame provided.
    # print(housing.describe(include= "all"))
    #Plotting histogram of the numerical attributes of the dataset using hist() method
    #* Reurns matplotlib.AxesSubplot or numpy.ndarray of them
    # hist = housing.hist(figsize = (15,10), bins = 50)
    # plt.show()
    #* Categorize the feature "median_income" using .cut() method of pandas. In this method, we categorize all entries between 0. to 1.5 as "1"
    income_cat = pd.cut(housing["median_income"], bins = [0., 1.5, 3.0, 4.5, 6.,20], labels = [1,2,3,4,5])
    #* Create an instance of the class StratifiedShuffleSplit() with random state for reproducibility 
    split_object = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    #* Generate indices to split data into training and test set. The arguments for slit are the training data and the training labels
    gen_obj = split_object.split(housing, income_cat)
    #* Calling next(gen_obj) once will provide the indices for the training and test sets in the variables train_ind and test_ind, respectively. 
    train_ind, test_ind = next(gen_obj)
    #* Now since we have the indices, we can use them to create a dataframe for training and testing
    strat_train_set = housing.loc[train_ind]
    strat_test_set = housing.loc[test_ind]
    #* We create a copy to play with, so that changes made do not affect the strat_train_set dataframe
    train_copy = strat_train_set.copy()
    """
    Plotting a scatter plot to visualize geographical distribution of median housing values
    The color of each point in the scatter plot is determined by the "median_house_value" 
    We set the alpha value to 0.4 to make the points semi-transparent, which helps visualize overlapping data points
    The colorbar parameter is set to True, displaying a color bar that maps the "median_house_value" to the color scale
    """
    train_copy.plot(kind="scatter", x="latitude", y="longitude", alpha= 0.4, figsize= (10,7), colorbar=True, cmap="jet", c="median_house_value", s=train_copy["population"]/100)
    # pd.plotting.scatter_matrix(train_copy[["latitude", "total_rooms", "median_income", "median_house_value"]])
    # plt.show()
    
    #* corr() is used to find the pairwise correlation of all columns in the Pandas Dataframe in Python. Any NaN values are automatically excluded.
    #* The corr() method is designed to work with numeric data.
    # print(train_copy.head(5))
    #! Todo: corr() after excluding the "ocean_proximity" column since its not numerical data
    # print(train_copy.corr())
    """
    The below elementwise division calculates the ratio of "total_rooms" to "median_house_value" for each row in the DataFrame
    This is done to add the new feature "rooms_per_household" that can potentially be more informative than the individual "total_rooms" and "median_house_value" columns
    Creating new meaningful features is a common practice in data analysis and machine learning
    """
    train_copy["rooms_per_household"] = train_copy["total_rooms"]/train_copy["median_house_value"]
    #* This new column "bedrooms_per_room" is to create another potentially informative feature.
    #* It represents the proportion of bedrooms relative to the total number of rooms
    train_copy["bedrooms_per_room"] = train_copy["total_bedrooms"]/train_copy["total_rooms"]
    #* Creation of another new feature
    train_copy["population_per_household"] = train_copy["population"]/train_copy["median_house_value"]
    """
    The purpose of creating housing_labels is to isolate the target variable (median house values) from the training 
    set for use in training and evaluating machine learning models. .copy() helps to extract a specific column so that any modfication
    done to this column does not affect the original dataframe strat_train_set. So we now have our target labels.
    """
    housing_labels = strat_train_set['median_house_value'].copy()
    #* Preparing the training data
    train_data = strat_train_set.drop("median_house_value", axis = 1)

    #------------------------------------------------------------------------------------------------------------------
    """Explanation:
    In real-world datasets, it's common to encounter missing values in various features or columns. 
    Missing data can cause issues during the training of machine learning models, as many algorithms 
    cannot handle missing values directly. The SimpleImputer class provides a convenient way to fill 
    in missing values with a specified strategy.
    """
    from sklearn.impute import SimpleImputer 
    #* With the "median" strategy, we are setting up the imputer to fill in missing values with the median of each column
    imputer = SimpleImputer(strategy = "median")
    #* Dropping the non-numerical data from train_data to perform imputation on
    housing_num = train_data.drop(train_data.select_dtypes(exclude=[np.number]), axis=1)
    """
    The .fit() method is used to calculate the median for each column with missing values in housing_num. 
    The .transform() method is then applied to impute the missing values with the calculated median values, 
    resulting in the housing_num_imputed NumPy array "out".
    The columns parameter is set to housing_num.columns, which means that the column names for the new DataFrame 
    will be the same as the column names from the housing_num DataFrame.
    """
    imputer.fit(housing_num)
    out = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(out, columns=housing_num.columns)
    
    from sklearn.preprocessing import OneHotEncoder 
    #* Create an instance of the OneHotEncoder Class
    encoder = OneHotEncoder()
    #* Since the ocean_proximity column of the train_data is not numeric, and is a categorical feature, we one hot encode it
    #* The fit_transform() method is called on the encoder object to perform the transformation.
    """
    The result of .fit_transform() operation is stored in the one_hot variable, which will contain a sparse matrix representation of 
    the one-hot encoded "ocean_proximity" column. A sparse matrix is used to efficiently represent datasets with many 
    zero entries (in this case, most entries are zeros in the one-hot encoded matrix). After one-hot encoding, each unique 
    category in the "ocean_proximity" column will be represented by a binary column in the one_hot matrix. The presence of 
    a category will be denoted by a 1, and the absence will be denoted by a 0. This transformation is often necessary for 
    machine learning algorithms that cannot directly handle categorical data and require numeric input.
    Keep in mind that the variable one_hot will be a sparse matrix, and we might want to convert it to a regular DataFrame 
    if we prefer a more traditional tabular representation. To do this, we can use the .toarray() method on the one_hot 
    sparse matrix
    """
    one_hot = encoder.fit_transform(train_data[['ocean_proximity']])
    print(one_hot)
    #------------------------------------------------------------------------------------------------------------------
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
    #* Create a custom class with the BaseEstimator and TransformerMixin as the base class
    class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
        #* This is a constructor method of our custom class. Meaning, this constructor function will be called
        #* when an instance of the class is created. The argument given is "add_bedrooms_per_room" and the default value
        #* is set as True. 
        def __init__(self,add_bedrooms_per_room= True):
            #* We then store the value of add_bedrooms_per_room in the instance variable
            self.add_bedrooms_per_room = add_bedrooms_per_room
        
            """
            This method is part of the TransformerMixin class, and it is typically used when a transformer needs to learn 
            something from the data during the training process. However, in this case, the CombinedAttributesAdder does not 
            need to learn any parameters during training. So, the method simply returns self, indicating that no fitting is required.
            """
        def fit(self,X, y=None):
            return self
            """
            This method is used to perform the actual transformation on the data. It takes the input data X as input and returns the 
            transformed data. Inside the transform() method, three new features are computed and added to the input data X
            """
        def transform(self, X, y=None):
            #* Calculated as the ratio of the total number of rooms to the total number of households
            rooms_per_household = X[:, 3] / X[:, 6]
            #* Calculated as the ratio of the total population to the total number of households
            population_per_household = X[:, 5] / X[:, 6]

            if self.add_bedrooms_per_room:
                #* This feature is conditionally computed only if add_bedrooms_per_room is set to True. It is calculated as the ratio 
                #* of the total number of bedrooms to the total number of rooms
                bedrooms_per_room = X[:, 4] / X[:, 3]
                #* This method returns the transformed data with the newly computed features appended using np.c_[]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

            else:
                return np.c_[X, rooms_per_household, population_per_household]
    #* Create an instance of the class we just defined
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    #* Call the transform method on the newly created object and set the add_bedrooms_per_room parameter as False
    housing_extra_attribs = attr_adder.transform(train_data.values)
    #------------------------------------------------------------------------------------------------------------------
    """
    The StandardScaler is used for data transformation that standardizes by subtracting the mean so the features are on the same scale
    and scaling them to unit variance. 
    """
    from sklearn.preprocessing import StandardScaler
    #* Create an instance of the StandardScaler class
    scaler = StandardScaler()
    #* Fit it on the data using the fit_transform() method
    scaled_features = scaler.fit_transform(housing_num)
    #------------------------------------------------------------------------------------------------------------------
    """
    We will now import the Pipeline Class from the pipeline module of sklearn
    This will helps us, as the name suggests, club up several sequential data processing or modelling steps into one
    sequence.
    Till now, we implemented three major data transformations: Imputation(imputer), Adding new features(attribs_adder)
    and the Standardization of the features(std_scaler)
    We will now chain all these data tansformations into a single pipeline as shown below
    """
    from sklearn.pipeline import Pipeline
    #* create an instance of the class and pass the three transformations as shown below
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = "median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler',  StandardScaler())
    ])
    #* Call the fit_transform method on the object num_pipeline, and pass the data that needs to be tansformed, after which
    #* save it in the variable housing_num_tr. The result stored will be a numpy array containing all the transormed features
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    #------------------------------------------------------------------------------------------------------------------
    """
    Now, bear in mind that using The pipeline class, we could only work on numerical columns in the data. But when we have to work on 
    the whole dataset, and if it contains both numerical and categorical data, we will have to use another transforming method that 
    can enable us to process our data sequentially, with both numerical and categorical types.
    For this purpose, we use the ColumnTransformer Class of the compose module of sklearn as follows
    """
    from sklearn.compose import ColumnTransformer
    #* Create an instance of the ColumnTransformer class and pass the following parameters
    full_pipeline = ColumnTransformer([
        #* num is the name of the first step, num_pipeline is the pipeline we created before that is the part of this step
        #* list(housing_num) specifies the list of numerical feature names that should be processed
        ("num", num_pipeline, list(housing_num)),
        #* cat is the name of the second step in this special pipeline, where we use the OneHotEncoder() to be performed on
        #* the "ocean_proximity" feature of the data, to convert the categorical feature to a numerical one
        ("cat", OneHotEncoder(), ["ocean_proximity"])
    ])
    #* Now fit this data transformation sequence to train_data(containing both the categorical and numerical data).
    #* Notice how we are not using the housing_num anymore, since it was only created to demonstrate the num_pipeine method
    housing_prepared = full_pipeline.fit_transform(train_data)
    #------------------------------------------------------------------------------------------------------------------
    """
    With this, we are done with the data processing part and now we can move on to the modelling part of our program
    The first predictor we will use is a Linear Regression (fitting a line to the data)
    Import the LinearRegression Class from the linear_model module of sklearn
    """
    from sklearn.linear_model import LinearRegression
    #* Create an instance of the Class
    lin_reg = LinearRegression()
    #* Fit this model on the training data that we got from the full_pipeline (fully pre-processed data) and the hosing_labels(target values)
    lin_reg.fit(housing_prepared, housing_labels)
    #* Now we will check the predictions of our model on the data it is trained on, by calling the method called .predct() on the training data
    predictions = lin_reg.predict(housing_prepared)
    # print(predictions, housing_labels)
    #------------------------------------------------------------------------------------------------------------------
    """
    Now we will import mean_squared_error function of the metrics module of sklearn
    The RMSE is a common metric for regression tasks that measures the average distance between the predicted and 
    actual values, and it is a more interpretable metric than the raw MSE
    However, it is important to note that evaluating the model on the training data alone can lead to optimistic results. It is generally 
    recommended to split the data into training and testing sets, train the model on the training set, and then evaluate it on the unseen test 
    set to obtain a more realistic performance estimate. This is commonly done using techniques like cross-validation to ensure a more robust 
    evaluation of the model's generalization capabilities.
    """
    from sklearn.metrics import mean_squared_error
    #* The parameters we feed to the function are the actual target values(housing_labels), predicted values by the model(predictions)
    #* and store the output in the variable lin_rmse
    #* Note that by setting the "squared" parameter to False, we compute the RMSE, otherise we will have the MSE
    lin_rmse = mean_squared_error(housing_labels, predictions, squared = False)

    """
    Now we will train another model(DecisionTreeRegressor) to make predictions on our data. Decision trees are one of the rule based methods.
    Import DecisionTreeRegressor Class from the module tree of sklearn
    """
    from sklearn.tree import DecisionTreeRegressor
    #* Create an instance of the class
    tree_reg = DecisionTreeRegressor()
    #* Fit the model on the training data
    tree_reg.fit(housing_prepared, housing_labels)
    #* Make predictions using the .predict() method and save it in the variable predictions
    predictions = tree_reg.predict(housing_prepared)
    #* Compute the RMSE using mean_squared_error() and setting the squared parameter to False, function and store it in tree_rmse
    tree_rmse = mean_squared_error(housing_labels, predictions, squared = False)
    """
    Now we will take a look at how to validate the predictions made by the model.
    We will import cross_val_score function of the model_selection module of the sklearn
    """
    from sklearn.model_selection import cross_val_score
    #* The cross_val_score function is called to perform cross-validation on the tree_reg model. It takes in the following arguments:
    #* tree_reg: The model to be cross validated
    #* housing_prepared: The dataset obtained from full-pipeline
    #* housing_labels: The target variable
    #* scoring="neg_root_mean_squared_error": The scoring parameter specifies the evaluation metric used for cross-validation. In this case, the negative root mean squared error (neg_root_mean_squared_error) is used
    #* cv=10: The cv parameter specifies the number of cross-validation folds. In this case, cross-validation is performed with 10 folds, meaning the dataset is divided into 10 parts, and the model is trained and evaluated 10 times, each time using a different fold as the validation set and the rest as the training set
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
    #* Store the absolute value of the scores in a variable
    scores = abs(scores)
    """
    We will now import the RandomForestRegressor Class from the ensemble module of sklearn
    This class is used for regression tasks and is an ensemble learning method that constructs multiple decision trees and averages their predictions.
    """
    from sklearn.ensemble import RandomForestRegressor
    #* Create an instance of the class
    forest_reg = RandomForestRegressor()
    #* Fi the model on the data
    forest_reg.fit(housing_prepared, housing_labels)
    #* Make predictions and store it in a variable
    predictions = forest_reg.predict(housing_prepared)
    #* Use the .mean_squared_error() function to compute RMSE
    forest_rmse = mean_squared_error(housing_labels, predictions, squared = False)
    #* Validate the score using the .cross_val_score() method
    scores = cross_val_score(forest_reg,housing_prepared, housing_labels,scoring = "neg_root_mean_squared_error", cv = 10)
    #* Storing the absolute value of the scores in a variable
    scores = abs(scores)
    print(scores)
    """
    From the model_selection module of sklearn, import GridSearchCV
    GridSearchCV to perform a grid search over hyperparameters for a random forest regression model (reg_forest). The grid 
    search helps to find the best combination of hyperparameters for the model that results in the lowest root mean squared error (RMSE) during cross-validation
    """
    from  sklearn.model_selection import GridSearchCV
    #* Create an instance of the class 
    reg_forest = RandomForestRegressor()
    #* The param_grid variable defines the grid of hyperparameters to search over. In this case, the grid includes the number of estimators (n_estimators) and the 
    #* maximum number of features to consider for splitting (max_features)
    param_grid = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
    #* The number of cross-validation folds (cv=5) is set to 5, meaning the dataset will be split into 5 parts for cross-validation
    grid_search = GridSearchCV(estimator=reg_forest, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=5)
    #* Fit it on our housing training data along with its labels 
    grid_search.fit(housing_prepared, housing_labels)
    #* Get the best parameters using the best_params_ attribute of the grid_search object 
    best_param = grid_search.best_params_
    #------------------------------------------------------------------------------------------------------------------
    """
    Now we will see the evaluation of the final model on the test set after performing the necessary preprocessing steps and obtaining the best estimator from the grid search
    """
    #* The labels of the test data
    y_test = strat_test_set['median_house_value'].copy()
    #* The test data based on which we will make predictions (After droping off the target values)
    X_test = strat_test_set.drop("median_house_value", axis = 1)
    #* We use the full pipeline that was created earlier and call the transform method on it, and pass the argument as the test data
    X_test_prepared = full_pipeline.transform(X_test)
    #* "final_model" is the object that holds the final best model we got after the grid search 
    final_model = grid_search.best_estimator_
    #* Using this best model, we will make the predictions, b y using the .predict() method on the transformed test data
    final_predictions = final_model.predict(X_test_prepared)
    #* Compute the RMSE to evaluate how our model performed on the test data 
    final_rmse = mean_squared_error(y_test, final_predictions, squared = False)
    print(final_rmse)