# Qualitative Activity Recognition of Weight Lifting Exercises

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:

Class A: exactly according to the specification
Class B: throwing the elbows to the front
Class C: lifting the dumbbell only halfway
Class D: lowering the dumbbell only halfway
Class E: throwing the hips to the front

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

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
We set the work dir and start with loading the required libraries. For futher reproducibility, we will set seeds before we create different models.

    # set work dir
    setwd("C:/Users/mdragt/SkyDrive/Coursera/ML")
    
    # load libraries
    library(caret)
    library(randomForest)
    library(rpart)
    library(rpart.plot)
    library(ggplot2)
    library(lattice)
    library(rattle)

### Getting the data
We Load the training and test data sets. We wil use the test set for the final validation

    # read training and testing data for coursera course. 
    trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    valUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    data <- read.csv(url(trainUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
    validation <- read.csv(url(valUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

### Cleaning and preparing the data
In this section, we will inspect, clean and prepare the data for further processing.

**Inspect the data**
We notice that the first columns don't contain usefull data and that there are many variables with missing data. Besides,some of the variables, like stddev_, avg_, kurtosis_ etc. seem to be derived from the original data. We will exclude them in the next step. Please notice that although the amount of columns of both training and validation set is equal, the last column of the validation set is different (problem_id) in comparison to the last column of the training set (classe). Make sure that you apply the same transformations on both your training and validation set.

    summary(data)
    summary(validation)
    dim(data)
    dim(validation)

**Remove unnecessary columns**

    # first 7 columns don't contain useful info
    data <- data[,-seq(1:7)]
    validation <- validation[,-seq(1:7)]

**Remove columns with NAs**
This reduces de amount of predictors to 53

    # select columns that don't have NAs
    indexNA<-as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
    data <- data[,!indexNA]
    validation <- validation[,!indexNA]

**Remove highly correlated variables**
Highly correlated variables can sometimes reduce the performance of a model, and will be excluded. However, this way of selection is disputable: http://arxiv.org/abs/1310.5726

    # set last (classe) and prior (- classe) column index
    last <- as.numeric(ncol(data))
    prior <- last - 1
    
    # set variables to numerics for correlation check, except the "classe"
    for (i in 1:prior) {
    data[,i] <- as.numeric(data[,i])
    validation[,i] <- as.numeric(validation[,i])
    }
    
    # check the correlations
    cor.check <- cor(data[, -c(last)])
    diag(cor.check) <- 0 
    plot( levelplot(cor.check, 
                    main ="Correlation matrix for all WLE features in training set",
                    scales=list(x=list(rot=90), cex=1.0),))
    
    # find the highly correlated variables
    highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
    
    # remove highly correlated variables
    data <- data[, -highly.cor]
    validation <- validation[, -highly.cor]

**Preproccesing of the variables**
The amount of predictors is now 46. We will continue with the preproccing of these predictors, by centering and scaling them

    # pre process variables
    preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
    dataPrep <- predict(preObj, data[,1:prior])
    dataPrep$classe <- data$classe
    
    valPrep <-predict(preObj,validation[,1:prior])
    valPrep$problem_id <- validation$problem_id


#### Remove the near zero variables

#### Create cross validation set

### Train Model 1: Random Forest
Set seed. Calculate optimal mtry.

### Results of Model 1
Plot OOB. Plot accuracy and Gini.



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
