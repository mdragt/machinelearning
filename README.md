# Qualitative Activity Recognition of Weight Lifting Exercises

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Goal

The goal of this project of the Coursera Practical Machine Learning course is to predict the manner in which people did the exercise. This is the "classe" variable in the training set.

## Report
This report describes:
+ how this model is built; 
+ how cross validation is used;
+ what the expected out of sample error is;
+ justification of the made choices;
+ Results of prediction model predicting 20 different test cases. 

### Reproducibility
Load libraries and set seed before the creation of different models

### Getting the data
Load the training and test set

### Cleaning and preparing the data
#### Inspect the data

#### Remove unnecessary columns 

#### Remove columns with NAs

#### Remove highly correlated variables

#### Preproccesing of the variables

#### Remove the near zero variables

#### Create cross validation set

### Train Model 1: Random Forest

### Results of Model 1


### Accuracy Model 1 on training set and cross validation set

### Train Model 2: Decision Tree

### Accuracy Model 2 on training set and cross validation set



## Data 
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3TROgwbfY

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

## Notes
This project is an assignment for the Coursera Practical Machine Learning course
