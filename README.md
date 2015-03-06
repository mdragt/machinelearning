# Qualitative Activity Recognition of Weight Lifting Exercises

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:

+ Class A: exactly according to the specification
+ Class B: throwing the elbows to the front
+ Class C: lifting the dumbbell only halfway
+ Class D: lowering the dumbbell only halfway
+ Class E: throwing the hips to the front

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
    indexNA <- as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
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
                    
![Correlation Matrix](/Images/CorMatrix.jpeg)

    # find the highly correlated variables
    highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
    
    # remove highly correlated variables
    data <- data[, -highly.cor]
    validation <- validation[, -highly.cor]

**Preproccesing of the variables**
The amount of predictors is now 46. We will continue with the preproccing of these predictors, by centering and scaling them. Remember that the last column of the validation set contained the problem_id.

    # pre process variables
    preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
    dataPrep <- predict(preObj, data[,1:prior])
    dataPrep$classe <- data$classe
    
    valPrep <-predict(preObj,validation[,1:prior])
    valPrep$problem_id <- validation$problem_id

**Remove the near zero variables**
Near zero variables have less prediction value, so we remove them. Although in this case, there are none to be elimnated.

    # test near zero variance
    myDataNZV <- nearZeroVar(dataPrep, saveMetrics=TRUE)
    if (any(myDataNZV$nzv)) nzv else message("No variables with near zero variance")
    dataPrep <- dataPrep[,myDataNZV$nzv==FALSE]
    valPrep <- valPrep[,myDataNZV$nzv==FALSE]

### Create a cross validation set
To train and test the model, we create a training and a testing set.

    # split dataset into training and test set
    inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
    training <- dataPrep[inTrain,]
    testing <- dataPrep[-inTrain,]

### Train Model 1: Random Forest
Regarding the data, we will expect Decision Tree and Random Forest to give the best results. We start with Random Forest. First we set a seed to make this project reproducable. We will use the tuneRF function to calculate the optimal mtry and use that in the random forest function.
   
    # set seed for reproducibility
    set.seed(12345)

    # get the best mtry
    bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
                       stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
    
    mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]
  
    # Model 1: RandomForest
    wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
                          keep.forest=TRUE, proximity=TRUE, 
                          importance=TRUE,test=testing)
    

### Results of Model 1
First we plot the Out-Of-Bag (OOB) error-rate. Besides, we will investigate the mean decrease of both accuracy and Gini score. As we can see it was correct to use 501 trees. 

    # plot the Out of bag error estimates
    layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
    par(mar=c(5,4,4,0)) #No margin on the right side
    plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
    par(mar=c(5,0,4,2)) #No margin on the left side
    plot(c(0,1),type="n", axes=F, xlab="", ylab="")
    legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)

![Out of bag error rate](/Images/OOB.jpeg)

    # plot the accuracy and Gini
    varImpPlot(wle.rf, main="Mean Decrease of Accuracy and Gini per variable")

![Accuracy and Gini scores](/Images/AccuracyGini.jpeg)
    
    # MDSplot (we couldn't execute this due to lack of memory)
    MDSplot(wle.rf, training$classe)

### Accuracy Model 1 on training set and cross validation set
Here we use Model 1 to predict both the training as the testing set. With the test set, we obtain an accuracy of 0.9951, which seems to be acceptable. However, we will also test the Decision Tree model.

    # results with training set
    predict1 <- predict(wle.rf, newdata=training)
    confusionMatrix(predict1,training$classe)
    
    Confusion Matrix and Statistics

    #              Reference
    # Prediction    A    B    C    D    E
    #          A 3906    0    0    0    0
    #          B    0 2658    0    0    0
    #          C    0    0 2396    0    0
    #          D    0    0    0 2252    0
    #          E    0    0    0    0 2525
    
    # Overall Statistics
                                     
    #            Accuracy : 1          
    #              95% CI : (0.9997, 1)
    # No Information Rate : 0.2843     
    # P-Value [Acc > NIR] : < 2.2e-16  
                                     
    #               Kappa : 1          
    # Mcnemar's Test P-Value : NA         

    # Statistics by Class:

    #                      Class: A Class: B Class: C Class: D Class: E
    # Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    # Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    # Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    # Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    # Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    # Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    # Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    # Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

    # results with test set
    predict2 <- predict(wle.rf, newdata=testing)
    confusionMatrix(predict2,testing$classe)
    
    # Confusion Matrix and Statistics

    #               Reference
    # Prediction    A    B    C    D    E
    #         A 1671    2    0    0    0
    #         B    2 1134   11    0    0
    #         C    0    3 1015    9    0
    #         D    0    0    0  954    0
    #         E    1    0    0    1 1082

    # Overall Statistics
                                          
    #             Accuracy : 0.9951          
    #               95% CI : (0.9929, 0.9967)
    #  No Information Rate : 0.2845          
    #  P-Value [Acc > NIR] : < 2.2e-16       
                                          
    #                Kappa : 0.9938          
    #  Mcnemar's Test P-Value : NA              

    # Statistics by Class:

    #                      Class: A Class: B Class: C Class: D Class: E
    # Sensitivity            0.9982   0.9956   0.9893   0.9896   1.0000
    # Specificity            0.9995   0.9973   0.9975   1.0000   0.9996
    # Pos Pred Value         0.9988   0.9887   0.9883   1.0000   0.9982
    # Neg Pred Value         0.9993   0.9989   0.9977   0.9980   1.0000
    # Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    # Detection Rate         0.2839   0.1927   0.1725   0.1621   0.1839
    # Detection Prevalence   0.2843   0.1949   0.1745   0.1621   0.1842
    # Balanced Accuracy      0.9989   0.9964   0.9934   0.9948   0.9998

    
### Train Model 2: Decision Tree

    # Model 2: Decision Tree
    dt <- rpart(classe ~ ., data=training, method="class")
    
    # fancyRpartPlot works for small trees, but not for ours
    fancyRpartPlot(dt)

![Decision Tree Plot](/Images/fancyPlot.jpeg)
    
### Accuracy Model 2 on training set and cross validation set
As we can see, this model is not improving the performance, having an accuracy of 0.6989. Therefor, we will continue with model 1.

    # cross validation
    predictDT <- predict(dt, testing, type = "class")
    confusionMatrix(predictDT, testing$classe)
    
    # Confusion Matrix and Statistics

    #               Reference
    #  Prediction    A    B    C    D    E
    #           A 1506  190   51   67   49
    #           B   53  616   59   92  184
    #           C   44  199  867  180  242
    #           D   67  114   46  616   99
    #           E    4   20    3    9  508

    # Overall Statistics
                                         
    #             Accuracy : 0.6989         
    #               95% CI : (0.687, 0.7106)
    #  No Information Rate : 0.2845         
    #  P-Value [Acc > NIR] : < 2.2e-16      
                                         
    #                Kappa : 0.618          
    # Mcnemar's Test P-Value : < 2.2e-16      

    # Statistics by Class:

    #                      Class: A Class: B Class: C Class: D Class: E
    # Sensitivity            0.8996   0.5408   0.8450   0.6390  0.46950
    # Specificity            0.9152   0.9182   0.8631   0.9338  0.99250
    # Pos Pred Value         0.8084   0.6135   0.5659   0.6539  0.93382
    # Neg Pred Value         0.9582   0.8928   0.9635   0.9296  0.89253
    # Prevalence             0.2845   0.1935   0.1743   0.1638  0.18386
    # Detection Rate         0.2559   0.1047   0.1473   0.1047  0.08632
    # Detection Prevalence   0.3166   0.1706   0.2603   0.1601  0.09244
    # Balanced Accuracy      0.9074   0.7295   0.8541   0.7864  0.73100
    

### Results
Finally, as the Random Forest model gave us the best result, we will apply that to our validation set and create the documents to submit.

    # Predict the class of the validation set
    answer<-predict(wle.rf,valPrep)
    answer
    
    # code as suggested by Coursera
    pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
    }
    
    pml_write_files(answer)

## Data 
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3TROgwbfY

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

## Notes
This project is an assignment for the Coursera Practical Machine Learning course
