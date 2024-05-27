import pandas as pd
import numpy as np

dt = pd.read_csv('.\\CreditPrediction.csv')

# Data preprocessing steps
# axis=1  ==> column
# axis=0  ==> row

dt.info()  # what we have?   10167 r + 20 c

# Remove duplicates
dt.drop_duplicates(inplace=True)

#inplace ==> changes submit on main data

# Remove rows with all elements as NaN(empty)
dt.dropna(how="all", inplace=True) # axis = 0  
# Remove columns with all elements as NaN(empty)  +  # Drop the 'Unnamed: 19'
dt.dropna(axis=1, how="all", inplace=True)

# Drop the 'CLIENTNUM' column
dt.drop("CLIENTNUM", axis=1, inplace=True)
dt.drop("Months_on_book", axis=1, inplace=True)
dt.drop("Total_Ct_Chng_Q4_Q1", axis=1, inplace=True)
# dt.replace("Unknown", np.NAN, inplace=True)


# # Drop the 'Unnamed: 19' column
# dt.drop("Unnamed: 19", axis=1, inplace=True)


row_sample = dt.iloc[2]
col_type = row_sample.apply(type)

for c, dd, in col_type.items():
    print(f"{c} : {dd}")  # we have str + float + int + ... = categorial & numberical

dt  # now what we have after clear?   no : CLIENTNUM  Unnamed: 19  (20-2)


#hot coding  ==> # r2 ==> 86%    mse ==> 11 m
# # Encoding categorical data
#converts categorical columns (whose values ​​are categorical) to dummy variables. 
# This causes each category to become a new column, and a value of 1 in that column indicates 

dt = pd.get_dummies(dt)
dt.info()

# import numpy as np
# # # #r2 ==> 83%    mse ==> 13 m
# Mapping_Gender = {
#     'F': 0,
#     'M': 20000,
# }

# Mapping_Education_Level = {
#     'Unknown': 0,
#     'Uneducated': 6,
#     'High School': 1,
#     'College': 2,
#     'Graguate': 3,
#     'Post-Graduate': 4,
#     'Doctorate': 5,
# }


# Mapping_Marital_Status = {
#     'Unknown': 0,
#     'Single': 4,
#     'Married': 8,
#     'Divorced': 1,
# }

# Mapping_Income_Category = {
#     'Unknown': 0,
#     'Less than $40k': 2,
#     '$40k - $60k': 5,
#     '$60k - $80k': 9,
#     '$80k - $120k': 1,
#     '$120k +': 7,
# }

# Mapping_Card_Category = {
#     'Blue': 1,
#     'Silver': 2,
#     'Gold': 3,
#     'Platinum': 4,
# }


# dt["Gender"] = dt["Gender"].map(Mapping_Gender)
# dt["Education_Level"] = dt["Education_Level"].map(Mapping_Education_Level)
# dt["Marital_Status"] = dt["Marital_Status"].map(Mapping_Marital_Status)
# dt["Income_Category"] = dt["Income_Category"].map(Mapping_Income_Category)
# dt["Card_Category"] = dt["Card_Category"].map(Mapping_Card_Category)


# row_sample = dt.iloc[1]
# col_type = row_sample.apply(type)
# for c, dd, in col_type.items():
#     print(f"{c} : {dd}") # now we have just numberical!

# dt[["Gender","Education_Level","Marital_Status","Income_Category","Card_Category"]]

dt.corr()

# dt.drop("Education_Level", axis=1, inplace=True)
# dt.drop("Education_Level_Doctorate", axis=1, inplace=True)
# dt.drop("Education_Level_Graduate", axis=1, inplace=True)
# dt.drop("Education_Level_High School", axis=1, inplace=True)
# dt.drop("Education_Level_Post-Graduate", axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, chi2


# Step 1: Split the dataset
X = dt.drop('Credit_Limit', axis=1)
y = dt['Credit_Limit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Step 2: Impute missing values
imputer = KNNImputer(n_neighbors=7, weights="uniform", metric="nan_euclidean")
X_train_filled = imputer.fit_transform(X_train)
X_train_filled = pd.DataFrame(X_train_filled, columns=X_train.columns)

# Step 3: Scale only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Step 4: Impute the missing values in the test data
X_test_filled = imputer.transform(X_test)
X_test_filled = pd.DataFrame(X_test_filled, columns=X_test.columns)

# Step 5: Scale the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test_filled)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Calculate Mutual Information
mi_scores = mutual_info_regression(X_train_scaled, y_train)

# Select features based on MI scores
# Here we assume a threshold of 0.01 for demonstration purposes
selected_features = X_train.columns[mi_scores > 0.01]
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]


# # Find the minimum value in X_train_selected
# min_value = np.min(X_train_selected)

# # Add the absolute value of min_value to all features
# X_train_non_negative = X_train_selected + np.abs(min_value)
# X_test_non_negative = X_test_selected + np.abs(min_value)

# # Assuming X_train_selected contains both numerical and categorical features
# # Select the top k features based on Chi-Square scores
# k = 5  # Choose the desired number of features
# selector = SelectKBest(chi2, k=k)
# X_train_chi2 = selector.fit_transform(X_train_non_negative, y_train)
# X_test_chi2 = selector.transform(X_test_non_negative)

# from sklearn.model_selection import train_test_split
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler

# # Step 1: Split the dataset
# temp = ["Income_Category", "Income_Category"]
# X = dt.drop('Credit_Limit', axis=1)
# y = dt['Credit_Limit']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 2: Impute missing values
# imputer = KNNImputer(n_neighbors=7, weights="uniform", metric="nan_euclidean")
# X_train_filled = imputer.fit_transform(X_train[temp])
# X_train_filled = pd.DataFrame(X_train_filled, columns=temp)

# # Step 3: Scale only the training data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_filled)
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# # Step 4: Impute the missing values in the test data
# X_test_filled = imputer.transform(X_test[temp])
# X_test_filled = pd.DataFrame(X_test_filled, columns=temp)

# # Step 5: Scale the test data using the scaler fitted on the training data
# X_test_scaled = scaler.transform(X_test_filled)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# import numpy as np
# from scipy import stats

# z_scores = np.abs(stats.zscore(X_train_selected))

# filtered_entries = (z_scores < 3).all(axis=1)
# features_clean = X_train_selected[filtered_entries]
# labels_clean = X_test_selected[filtered_entries]

# # Calculate IQR and bounds
# Q1 = X_train_scaled.quantile(0.3)
# Q3 = X_train_scaled.quantile(0.7)
# IQR = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers in X_train_scaled
# outliers_mask = ((X_train_scaled < lower_bound) | (X_train_scaled > upper_bound)).any(axis=1)

# # Filter X_train_scaled and y_train
# X_train_scaled_no_outliers = X_train_scaled[~outliers_mask]
# y_train_no_outliers = y_train.loc[X_train_scaled.index[~outliers_mask]]

# # Repeat for X_test_scaled and y_test
# outliers_mask_test = ((X_test_scaled < lower_bound) | (X_test_scaled > upper_bound)).any(axis=1)
# X_test_scaled_no_outliers = X_test_scaled[~outliers_mask_test]
# y_test_no_outliers = y_test.loc[X_test_scaled.index[~outliers_mask_test]]

# Q1 = X_train_scaled['Credit_Limit'].quantile(0.3)
# Q3 = X_train_scaled['Credit_Limit'].quantile(0.7)
# IQR = Q3 - Q1

# X_train_scaled = X_train_scaled[~((X_train_scaled['Credit_Limit'] < (Q1-1.5*IQR)) | (X_train_scaled['Credit_Limit'] > (Q3+1.5*IQR)))]


# Model selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=11,
                              min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                              max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                              bootstrap=True, oob_score=False, n_jobs=None, random_state=30,
                              verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

model.fit(X_train_selected, y_train)

# Apply Recursive Feature Elimination (RFE)
rfe = RFE(model, n_features_to_select=17)  # Choose the desired number of features
X_train_rfe = rfe.fit_transform(X_train_selected, y_train)
X_test_rfe = rfe.transform(X_test_selected)

y_pred = model.predict(X_test_rfe)

#----------------------------------------------------------------------------------------
# from sklearn.ensemble import AdaBoostRegressor

# # Initialize AdaBoostRegressor   ==>   MSE = 10 mil    not good!
# adaboost_model = AdaBoostRegressor(base_estimator=model, n_estimators=100, random_state=42)

# # Fit the model with RFE-transformed features
# adaboost_model.fit(X_train_rfe, y_train)

# # Predictions
# y_pred_adaboost = adaboost_model.predict(X_test_rfe)

# # Evaluate the performance (you can use any relevant metric)
# # For example, mean squared error (MSE):
# from sklearn.metrics import mean_squared_error
# mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
# print(f"AdaBoost MSE: {mse_adaboost:.4f}")

# Model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

scores = cross_val_score(model, X_train_filled, y_train, cv=5)
print("Cross-validated scores:", scores)

#--------------------------------------------------------------

r2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print(f"R2 error:", {r2})
print(f"Mean Squared Error(MSE):", {MSE})
print(f"R Mean Squared Error(RMSE):", {math.sqrt(MSE)})
print(f"Mean Absolute Error(MAE):", {MAE})

print(y_pred)
print(y_test)
print(y_train)

import matplotlib.pyplot as plt

# test ==> actual   /    pred ==> predicted
leftovers = y_test - y_pred
#same as
plt.figure(figsize=(5,2.5))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.title("Y_Test Vs. Y_Pred")
plt.xlabel("Actual Credit_Limit")
plt.ylabel("Predicted Credit_Limit")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.show()

#---------------------------------------------------------------------------------------
#errors
plt.figure(figsize=(5,2.5))
plt.scatter(y_pred, leftovers, alpha=0.4)
plt.title("LeftOvers")
plt.xlabel("Predicted Credit_Limit")
plt.ylabel("leftovers")
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="black", linestyles="--")
plt.show()

