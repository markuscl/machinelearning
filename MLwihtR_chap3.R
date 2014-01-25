#Machine Learning with R
#Replicating Diagnostic breast cancer with the kNN algorithm Chapter3



#step1 collecting the data
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

wbcd <- read.csv(url, stringsAsFactors = FALSE)
names(wbcd) <- c("id", "diagnosis"  ,
                 "radius_mean"      , "texture_mean"       , "perimeter_mean"  ,
                 "area_mean"        , "smoothness_mean"    , "compactness_mean", 
                 "concavity_mean"   , "concavepoints_mean" , "symmetry_mean", "fractaldim_mean",
                 
                 "radius_sd"      , "texture_sd"       , "perimeter_sd"  ,
                 "area_sd"        , "smoothness_sd"    , "compactness_sd", 
                 "concavity_sd"   , "concavepoints_sd" , "symmetry_sd", "fractaldim_sd",
                 
                 "radius_max"      , "texture_max"       , "perimeter_max"  ,
                 "area_max"        , "smoothness_max"    , "compactness_max", 
                 "concavity_max"   , "concavepoints_max" , "symmetry_max", "fractaldim_max" )


#step2 exploring and preparing the data

#deleting id
wbcd <- wbcd[-1]
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B","M"),
                         labels = c("Benign", "Malignant"))

round(prop.table(table(wbcd$diagnosis)) * 100, digits =1)

summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])


normalize <- function (x) {
             return ( (x- min(x))/(max(x) - min(x)))
}


wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$area_mean)

wbcd_train <- wbcd_n[1:469,]
wbcd_test  <- wbcd_n[470:568,]

wbcd_train_labels <- wbcd[1:469,1]
wbcd_test_labels <- wbcd[470:568,1]


#step3 training a model on the data
library(class)

wbcd_test_pred <- knn(train=wbcd_train, test =wbcd_test,
                      cl= wbcd_train_labels, k=21)


#step4 evaluating model performance
CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=FALSE)

#step5 improving model performance
wbcd_z <- as.data.frame(scale(wbcd[-1]))

summary(wbcd_z$area_mean)
wbcd_train <- wbcd_z[1:469,]
wbcd_test  <- wbcd_z[470:568,]
wbcd_test_pred <- knn(train=wbcd_train, test =wbcd_test,
                      cl= wbcd_train_labels, k=21)
CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=FALSE)



