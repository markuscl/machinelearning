#Machine Learning with R
#Replicating Filtering mobile phone spam with the Naive Bayes Chapter4
library(tm)
library(gmodels)
require(wordcloud)


#step1 collecting data
#reading sms raw data
sms_raw <-RWeka :: read.arff("P:/R/MachineLearning/WithR/NaiveBayes/smsspamcollection.arff")

#step2 exploring and preparing the data


str(sms_raw)

sms_raw$class <- factor(sms_raw$class)
table(sms_raw$class)
sms_corpus <- Corpus(VectorSource(sms_raw$text))

corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removeWords, c("you","not","the", "and","for","that","are", "your"))
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_raw_train <- sms_raw[1:4169,]
sms_raw_test  <- sms_raw[4170:5568,]

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test  <- sms_dtm[4170:5568,]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5568]

prop.table(table(sms_raw_train$class))
prop.table(table(sms_raw_test$class))

wordcloud(sms_corpus_train, min.freq=40, random.order=FALSE)

spam <- subset(sms_raw_train, class == "spam")
ham  <- subset(sms_raw_train, class == "ham")


wordcloud(spam$text, max.words=40, scale=c(3,0.5))
wordcloud(ham$text, max.words=40, scale=c(3,0.5))

sms_dict <- findFreqTerms(sms_dtm_train,5)

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary=sms_dict))


convert_counts <- function (x) {
  x <- ifelse(x > 0, 1 , 0)
  x <- factor(x, levels=c(0,1), labels=c("No","Yes"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN=2, convert_counts)
sms_test  <- apply(sms_test , MARGIN=2, convert_counts)

#step3 training a model on the data
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$class)


#step4 evaluating model performance
sms_test_pred  <- predict(sms_classifier, sms_test)
CrossTable(sms_test_pred, sms_raw_test$class, prop.chisq = FALSE, prop.t = FALSE, prop.r=FALSE,
           dnn = c("predicted","actual"))

#step5 improving model performance
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$class, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$class, prop.chisq = FALSE, prop.t = FALSE, prop.r=FALSE,
           dnn = c("predicted","actual"))


