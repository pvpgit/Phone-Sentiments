##Read the Samsang Galaxy Data
Galaxy<- read.csv(paste("~/Desktop/Project_3/GalaxyLargeMatrix.csv", sep=""), header = TRUE)

#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
#install.packages("corrplot")
library(corrplot)

#Basic Commands
#No. of rows
nrow(Galaxy) #12877
#No. of columns
ncol(Galaxy) #60
# Range
Range<-range(Galaxy$samsunggalaxySentiment)  #-140 997

#will return the positions of the variables that are flagged to be problematic
nzv<-nearZeroVar(Galaxy)
filteredDescr <-Galaxy[,-nzv]
dim(filteredDescr) #12877 10
str(filteredDescr)
filteredDescr$id<-NULL

#Create object containing correlation results
descrCor <-  cor(filteredDescr)

#Summary 
summary(descrCor[upper.tri(descrCor)])
corrplot(descrCor)

####### mydata attributes = 19
# As I am working with Galaxy Sentiment, I will use all the samsung variables
df_Galaxy=Galaxy[,grepl("^samsung*",names(Galaxy))]
df_Galaxy$samsunggalaxy<-NULL

# I added the variables I got in descrCor to the above df to create 
#the list of variables that I feel are meaningful for this excercise
Galaxydata<- cbind(filteredDescr,df_Galaxy)
head(df_Galaxy)
head(Galaxydata)
ncol(Galaxydata)
foo<-cor(Galaxydata)
corrplot(foo)
#####Plots and Graphs
ggplot(Galaxydata,
       aes(
         y=samsunggalaxySentiment,
         x=samsunggalaxy
       )
) + geom_point() + stat_smooth(method ="lm", se = F)

ggplot(Galaxydata, aes(x=samsungdispos, y=samsungdisneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("Samsung Galaxy Display")

ggplot(Galaxydata, aes(x=samsungcampos, y=samsungcamneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("Samsung Galaxy Camera")

ggplot(Galaxydata, aes(x=samsungperpos, y=samsungperneg)) + 
  geom_point() + 
  geom_smooth()+
  ggtitle("Samsung Galaxy Performance")

# create a df of sentiment only (for histogram)
sentimentFactor <- data.frame(as.numeric(Galaxydata[,ncol(Galaxydata)]))
# remove zeroes
sentimentFactorNonZero <- subset(data.frame(sentimentFactor), sentimentFactor != 0)

# remove outliers (> +/- 3 stdevs)
sentimentFactorNonZero <- (
  subset(
    data.frame(sentimentFactorNonZero),
    sentimentFactorNonZero < 2*sd(sentimentFactorNonZero[,1]) &
      sentimentFactorNonZero > -2*sd(sentimentFactorNonZero[,1])
  )
)

# reset the df index, needed for charting
sentimentFactorNonZero <- data.frame(sentimentFactorNonZero[,1])

ggplot(sentimentFactorNonZero,
       aes(
         x = data.frame(sentimentFactorNonZero)
       )
) + geom_histogram(binwidth = 10) +
  xlab("Galaxy sentiment non zero values") +
  ggtitle("Galaxy Sentiment")

dev.off()

##################################
install.packages("arules")
library(arules)

##Discretize Samsung Galaxy Sentiment and create a vector containing discretized levels
samsung_disfixed7 <- discretize(Galaxydata$samsunggalaxySentiment, "fixed", 
                                categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))

ggp<-ggplot(data.frame(samsung_disfixed7),aes(x=samsung_disfixed7))
ggp + geom_histogram(fill="lightgreen",  stat="count") +
  xlab("Galaxy sentiment discretized values") +
  ggtitle("Galaxy Sentiment") +
  scale_y_log10()

summary(samsung_disfixed7)
str(samsung_disfixed7)

#Insert the vector into the dataset
Galaxydata$samsunggalaxySentiment <- samsung_disfixed7

set.seed(170)
##Create Data Partition
ptr=0.7
inTrain <- createDataPartition(y = Galaxydata$samsunggalaxySentiment, p=ptr, list=FALSE)

Galaxy_training <- Galaxydata[ inTrain,]
Galaxy_testing <- Galaxydata[-inTrain,]
nrow(Galaxy_training)
nrow(Galaxy_testing)

##trainControl
ctrl <- trainControl(method = "repeatedcv", number=10, repeats = 1)

######## KNN ############
##train                   
GalaxyknnFit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "knn",
                      trControl=ctrl, preProcess = "scale")
GalaxyknnFit

##predict
GalaxyknnPredict <- predict(GalaxyknnFit,Galaxy_testing)
table(GalaxyknnPredict)

##Confusion Matrix
confusionMatrix(GalaxyknnPredict, Galaxy_testing$samsunggalaxySentiment)

##postResample
print(postResample(pred=GalaxyknnPredict, obs=as.factor(Galaxy_testing[,"samsunggalaxySentiment"])))

######### Random Forest ##########
install.packages("randomForest")
library(randomForest)

##Random Forest Model 10 fold cross validation
rfFit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "rf",
               tuneLength = 10, trControl=ctrl, preProcess = c("center", "scale"))

rfFit

##predict
GalaxyrfPredict <- predict(rfFit, newdata = Galaxy_testing)
table(GalaxyrfPredict)

##Confusion Matrix
confusionMatrix(GalaxyrfPredict, Galaxy_testing$samsunggalaxySentiment)

##Resample
print(postResample(pred=GalaxyrfPredict, obs=as.factor(Galaxy_testing[,"samsunggalaxySentiment"])))

resampleHist(rfFit)
###### SVM ############
#install.packages("e1071")
library("e1071")

GalaxysvmFit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "svmRadial",
                      tuneLength = 10, trControl=ctrl, preProcess = c("center", "scale"))
GalaxysvmFit
grid <- expand.grid(sigma = c(.1,.2,.3),
                    C = c(1,2,4))

GalaxysvmFit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "svmRadial",
                      tuneGrid=grid, trControl=ctrl, preProcess = c("center", "scale"))
GalaxysvmFit

##predict
GalaxysvmPredict <- predict(GalaxysvmFit, newdata = Galaxy_testing)
table(GalaxysvmPredict)

##Confusion Matrix
confusionMatrix(GalaxysvmPredict, Galaxy_testing$samsunggalaxySentiment)

##postResample
print(postResample(pred=GalaxysvmPredict, obs=as.factor(Galaxy_testing[,"samsunggalaxySentiment"])))

######## C5.0 ########
install.packages("C50")
library(C50)
library(mlbench)

GalaxyC50Fit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "C5.0",
                      tuneLength = 10, trControl=ctrl)
GalaxyC50Fit
GalaxyC50grid <- expand.grid( .winnow = FALSE, .trials=c(10,20,30), .model=c("tree"))
GalaxyC50Fit <- train(samsunggalaxySentiment ~ ., data = Galaxy_training, method = "C5.0",
                      tuneGrid=GalaxyC50grid, trControl=ctrl)

##predict
GalaxyC50Predict <- predict(GalaxyC50Fit, Galaxy_testing)
table(GalaxyC50Predict)

##Confusion Matrix
confusionMatrix(GalaxyC50Predict, Galaxy_testing$samsunggalaxySentiment)

##postResample
print(postResample(pred=GalaxyC50Predict, obs=as.factor(Galaxy_testing[,"samsunggalaxySentiment"])))
