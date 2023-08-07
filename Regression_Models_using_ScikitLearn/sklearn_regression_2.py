# Predicting bike rental prices using Sklearn regression models, numpy, and pandas libraries
#! Todo: Documenting the process
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
#-----------------------------------------------------

path = "./bikes.csv"
bikesData = pd.read_csv(path)
bikesData.info()
bikesData.head(5)
mean = bikesData["hum"].mean()
columnsToDrop = ["instant", "casual", "registered", "atemp", "dteday"]
bikesData = bikesData.drop(columnsToDrop, axis = 1)
#-----------------------------------------------------
np.random.seed(42)
from sklearn.model_selection import train_test_split
bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24
train_set, test_set = train_test_split(bikesData, train_size= 0.7, test_size= 0.3, random_state = 42)
train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)
print("Number of instances in train_set:", len(train_set))
print("Number of instances in test_set:", len(test_set))
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
#-----------------------------------------------------
columnsToScale = ['temp', 'hum', 'windspeed']
scaler = StandardScaler()
train_set[columnsToScale] = scaler.fit_transform(train_set[columnsToScale])
test_set[columnsToScale] = scaler.transform(test_set[columnsToScale])
#-----------------------------------------------------
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
trainingLabels = train_set["cnt"].copy()
trainingCols = train_set.drop(["cnt"], axis=1)
#-----------------------------------------------------
dec_reg = DecisionTreeRegressor(random_state =  42)
dt_mae_scores = -cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(dt_mae_scores)
dt_mse_scores = np.sqrt(-cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))
display_scores(dt_mse_scores)
#-----------------------------------------------------
lin_reg = LinearRegression()
lr_mae_scores = -cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(lr_mae_scores)
lr_mse_scores = np.sqrt(-cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))
display_scores(lr_mse_scores)
#-----------------------------------------------------
forest_reg = RandomForestRegressor(random_state = 42, n_estimators=150)
rf_mae_scores = -cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(rf_mae_scores)
rf_mse_scores = np.sqrt(-cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))
display_scores(rf_mse_scores)
#-----------------------------------------------------

#-----------------------------------------------------
from sklearn.model_selection import GridSearchCV
param_grid = [
    {"n_estimators": [120, 150], 
    "max_features": [10,12],
    "max_depth": [15,28]},
             ]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')
grid_search.fit(trainingCols, trainingLabels)
best_estimator = grid_search.best_params_
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

best_parameters = grid_search.best_params_
best_n_estimators = best_parameters['n_estimators']
best_value = best_parameters['max_features']
max_depth = best_parameters["max_depth"]
print(max_depth)
#-----------------------------------------------------
final_model = grid_search.best_estimator_
test_set.sort_values("dayCount", axis= 0, inplace=True)
test_y_cols = "cnt"
test_x_cols = test_set.drop(["cnt"], axis=1).columns.values
X_test = test_set.loc[:,test_x_cols]
y_test = test_set.loc[:,test_y_cols]
#-----------------------------------------------------
test_set.loc[:,"predictedCounts_test"] = final_model.predict(X_test)
mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])
final_mse = np.sqrt(mse)
print(final_mse)
times = [9,18]
for time in times:
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    test_set_freg_time = test_set[test_set.hr == time]
    test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'cnt', ax = ax)
    test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'predictedCounts_test', ax =ax)
    plt.show()

