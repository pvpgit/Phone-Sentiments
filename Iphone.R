##Read the IPhone Data
iPhone<- read.csv(paste("~/Desktop/Project_3/iPhoneLargeMatrix.csv", sep=""), header = TRUE)

#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
#install.packages("corrplot")
library(corrplot)

#Basic Commands
##No. of rows
nrow(iPhone)
##No. of columns
ncol(iPhone)
## Range
range(iPhone$iphoneSentiment)

dim(iPhone) ##60 attributes 12877 instances
nzviphone<-nearZeroVar(iPhone)
filteredDescr_iphone <-iPhone[,-nzviphone]
dim(filteredDescr_iphone) #12877 10
str(filteredDescr_iphone)
filteredDescr_iphone$id<-NULL

#Create object containing correlation results
descrCor_iphone <-  cor(filteredDescr_iphone)

#Summary 
summary(descrCor_iphone[upper.tri(descrCor_iphone)])
corrplot(descrCor_iphone)

####### mydata attributes = 15 
# As I am working with IPhone Sentiment, I will use all the iphone variables
iphone_df=iPhone[,grepl("^iphone*",names(iPhone))]
iphone_df$iphone<-NULL
iphone_df$iphonedispos<- NULL
iphone_df$iphonedisneg<- NULL
iphone_df$iphonedisunc<- NULL
iphone_df$iphoneperpos<- NULL

# I added the variables I got in descrCor to the above df to create 
#the list of variables that I feel are meaningful for this excercise
iphonedata<- cbind(filteredDescr_iphone,iphone_df)
head(iphone_df)
head(iphonedata)
ncol(iphonedata)
foo1<- cor(iphonedata)
corrplot(foo1)

#####Plots and Graphs
ggplot(iphonedata,
       aes(
         y=iphoneSentiment,
         x=iphone
       )
) + geom_point() + stat_smooth(method ="lm", se = F)

ggplot(iphonedata, aes(x=iphonedispos, y=iphonedisneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("iPhone Display")

ggplot(iphonedata, aes(x=iphonecampos, y=iphonecamneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("iPhone Camera")

ggplot(iphonedata, aes(x=iphoneperpos, y=iphoneperneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("iPhonePerformance")

# create a df of sentiment only (for histogram)
iphone_sentimentFactor <- data.frame(as.numeric(iphonedata[,ncol(iphonedata)]))
# remove zeroes
iphone_sentimentFactorNonZero <- subset(data.frame(iphone_sentimentFactor), iphone_sentimentFactor != 0)

# remove outliers (> +/- 3 stdevs)
iphone_sentimentFactorNonZero <- (
  subset(
    data.frame(iphone_sentimentFactorNonZero),
    iphone_sentimentFactorNonZero < 2*sd(iphone_sentimentFactorNonZero[,1]) &
      iphone_sentimentFactorNonZero > -2*sd(iphone_sentimentFactorNonZero[,1])
  )
)

# reset the df index, needed for charting
iphone_sentimentFactorNonZero <- data.frame(iphone_sentimentFactorNonZero[,1])

ggplot(iphone_sentimentFactorNonZero,
       aes(
         x = data.frame(iphone_sentimentFactorNonZero)
       )
) + geom_histogram(binwidth = 10) +
  xlab("Iphone sentiment non zero values") +
  ggtitle("Iphone Sentiment")

dev.off()

#install.packages("arules")
library(arules)
##Discretize iphone Sentiment and create a vector containing discretized levels
iphone_disfixed7 <- discretize(iphonedata$iphoneSentiment, "fixed", 
                               categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))

ggp<-ggplot(data.frame(iphone_disfixed7),aes(x=iphone_disfixed7))
ggp + geom_histogram(fill="lightgreen",  stat="count") +
  xlab("iphone sentiment discretized values") +
  ggtitle("iPhone Sentiment") +
  scale_y_log10()

#Insert the vector into the dataset
#iphonedata$iphoneSentiment <- iphone_disfixed7

set.seed(107)
##Create Data Partition
ptr=0.7
iphone_inTrain <- createDataPartition(y = iphonedata$iphoneSentiment, p=ptr, list=FALSE)

iphone_training <- iphonedata[iphone_inTrain,]
iphone_testing <- iphonedata[-iphone_inTrain,]
nrow(iphone_training)
nrow(iphone_testing)

##trainControl
iphone_ctrl <- trainControl(method = "repeatedcv", number=10, repeats = 1)

######## KNN ############
##train  
iPhoneknnFit <- train(iphoneSentiment ~ ., data = iphone_training, method = "knn",
                      trControl=iphone_ctrl, preProcess = "scale")
iPhoneknnFit

##predict
iphoneknnPredict <- predict(iPhoneknnFit, newdata = iphone_testing)
table(iphoneknnPredict)

##Confusion Matrix
confusionMatrix(iphoneknnPredict, iphone_testing$iphoneSentiment)

##postResample
print(postResample(pred=iphoneknnPredict, obs=as.factor(iphone_testing[,"iphoneSentiment"])))

######### Random Forest ##########
#install.packages("randomForest")
library(randomForest)

##Random Forest Model 10 fold cross validation
iphonerfFit <- train(iphoneSentiment ~ ., data = iphone_training, method = "rf",
                     tuneLength = 10, trControl=iphone_ctrl, preProcess = c("center", "scale"))
iphonerfFit

##predict
iphonerfPredict <- predict(iphonerfFit, newdata = iphone_testing)
table(iphonerfPredict)

##Confusion Matrix
confusionMatrix(iphonerfPredict, iphone_testing$iphoneSentiment)

##Resample
print(postResample(pred=iphonerfPredict, obs=as.factor(iphone_testing[,"iphoneSentiment"])))

###### SVM ############
install.packages("e1071")
library("e1071")

iphonesvmFit <- train(iphoneSentiment ~ ., data = iphone_training, method = "svmRadial",
                      tuneLength = 10, trControl=iphone_ctrl, preProcess = c("center", "scale"))
iphonesvmFit
iphonegrid <- expand.grid(sigma = c(0.4, 0.6, 0.7),
                          C = c(6,8,10))

iphonesvmFit <- train(iphoneSentiment ~ ., data = iphone_training, method = "svmRadial",
                      tuneGrid=iphonegrid, trControl=iphone_ctrl, preProcess = c("center", "scale"))

iphonesvmFit

##predict
iphonesvmPredict <- predict(iphonesvmFit, newdata = iphone_testing)
table(iphonesvmPredict)

##Confusion Matrix
confusionMatrix(iphonesvmPredict, iphone_testing$iphoneSentiment)

##postResample
print(postResample(pred=iphonesvmPredict, obs=as.factor(iphone_testing[,"iphoneSentiment"])))

######## C5.0 ########
install.packages("C50")
library(c50)
library(mlbench)

iphoneC50Fit <- train(iphoneSentiment ~ ., data = iphone_training, method = "C5.0",
                      tuneLength = 10, trControl=iphone_ctrl)
iphoneC50Fit
iphonegrid <- expand.grid( .winnow = c(TRUE), .trials=c(80,90), .model=c("rules"))
iphoneC50Fit <- train(iphoneSentiment ~ ., data = iphone_training, method = "C5.0",
                      tuneGrid=iphonegrid, trControl=iphone_ctrl)

##predict
iphoneC50Predict <- predict(iphoneC50Fit, newdata = iphone_testing)
table(iphoneC50Predict)

##Confusion Matrix
confusionMatrix(iphoneC50Predict, iphone_testing$iphoneSentiment)

##postResample
print(postResample(pred=iphoneC50Predict, obs=as.factor(iphone_testing[,"iphoneSentiment"])))
