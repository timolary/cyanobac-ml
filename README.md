# Predicting Cynobacteria Levels using Machine Learning 
----
### Mahalia Clark and Tim Laracy


Cyanobacteria blooms are a major problem in Lake Champlain. We have applied a variety of machine learning methods to a unique, public, long-term dataset on cyanobacteria levels and other water quality variables produced by the VT Department of Environmental Conservation. We analyze ten years of data from 15 sampling locations around Lake Champlain with the goal of predicting cyanobacteria levels from other water quality indicators.


Files and folders in this Directory:


data/

Folder containing the data file: AllSites.csv


figures/

Folder containing generated figures.


NN_weights/





data_manager.py

Python file with a get_data() function to import a pre-processed dataframe.



Cyanobacteria_DataExploration.ipynb

Jupyter notebook where the data is imported and explored. Rows with missing cyanobacteria levels are dropped. A target column is created with 1 for cyanobacteria levels over 4e8, 0 otherwise. A histogram is created showing the distribution of cyanobacteria levels. The dataset is cleaned, removing stray letters from numeric values, removing unused columns, creating a new feature with the sampling month of each observation, and a new column with the molar N:P ratio calculated from the total P and total N features. Missing values in the features are filled in with the monthly average for that feature and sampling month, calculated accross all years. A separate dataframe is created with the targets for the next sample time for each location, however due to the large inconsistancies in sampling frequency (1 week to many weeks) this was not used in the end. Finally, PCA visualizations are created in 2D and 3D.


Cyanobacteria_LinearRegression.ipynb

Jupyter notebook with code for linear and logistic regression with and without polynomial features. Data is imported, split (80:20), and X's and y's are created for linear regression (unscaled X) and logistic regression (scaled X), with and without polynomial features. Linear regression is run with 5-fold cross validation and testing with and without polynomial features. It is also run on all the data to get coefficients and their significance levels. Logistic regression is run with 5-fold cross validation and testing with and without polynomial features. RandomizedSearchCV is used to optimize hyperparameters for logistic regression with and without hyperparameters however this yields worse results so we stick with the hand-tuned models.


Cyanobacteria_RandomForest.ipynb

Data is imported and split (80:20). Random Forest and Extra Trees classifiers are run with 5-fold cross-validation and testing, each with default parameters and 500 trees. Random Forest Regression is also run with default parameters and 500 trees, using out of bag scores for validation as well as testing on the test set. Hyperparameters are also tuned for each of the three models using RandomizedSearchCV however the resulting models performed the same or worse as the basic models so we stick with the basic ones. Feature importance is printed for each of the three models.


Cyanobacteria_SVM.ipynb




Cyanobacteria_NeuralNetwork.ipynb



