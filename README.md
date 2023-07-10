# Missing Value Prediction with Intelligent Supervised Learning Algorithms and Hyperparameter optimization

This repo contains a tool created to clean categorical data tables with the random forest decision tree algorithm and genetic hyperparameter optimization to find optimal hypeprparameters automatically. 

In order to use the tool two asumptions have to be checked. The algorithm uses columns in the dataset to predict missing values in other columns. Therefore enough non missing data has to be provided in the data table to build the classifier. The data columns furthermore have to be correlated to get valid predictions for a column in question from other columns as features. 

The full_tool.py script contains the dataCleaner function which is a high level wrapper for the full tool. Here functionalities like the additional creation of a plot overviewing missing values in the dataset or a plot showing corrleations to the to be predicted column can be created. After specification of all the arguments of the function the tool takes the categorical data table, recodes the data into scalar number format for model training, uses all available data to train a model for the column in question with automated genetic hyperparameter optimization and predicts the unknown missing values.

The test_script.py script gives an overview of how to use the tool for a full data table automtically. The folder structure has to be used as given in the repo for the tool to work correctly, otherwise the tool has to be adapted. The tool also gives the option to save training data and trained models in order to reuse them for further use cases.

