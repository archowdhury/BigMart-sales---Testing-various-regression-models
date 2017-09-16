library(dplyr)
library(zoo)
library(caret)
library(qdap)

RMSE = function(predicted, actual)
{
  rmse = sqrt(mean((predicted-actual)^2))
  return(rmse)
}


#=======================================================#
# Read and format the training data                     #
#=======================================================#

train = read.csv("C:/Users/amit.r/Desktop/Big Mart Sales/Train.csv", stringsAsFactors = FALSE)
test  = read.csv("C:/Users/amit.r/Desktop/Big Mart Sales/Test.csv", stringsAsFactors = FALSE)

# Combine the train and test datasets to do feature engineering
train$Source = "train"
test$Source = "test"
all = bind_rows(train, test)

# Check for NA's
all$Outlet_Size[all$Outlet_Size==""] = NA
sapply(all,function(x) sum(is.na(x)))

# Check for unique values
sapply(all, function(x) length(unique(x)))


# Correct the Fat Content
table(all$Item_Fat_Content)
all$Item_Fat_Content[grep("L|ow", all$Item_Fat_Content)] = "Low Fat"
all$Item_Fat_Content[grep("R|r", all$Item_Fat_Content)] = "Regular"


# Impute Item Weight by assigning the mean weight for each Item_Identifier
all = transform(all,Item_Weight = na.aggregate(Item_Weight, by=Item_Identifier))
all %>% arrange(Item_Identifier) %>% select(Item_Identifier, Item_Weight) %>% head(20)

# Impute missing values for Outlet Size
table(all$Outlet_Size , all$Outlet_Type, useNA="always")
all$Outlet_Size[is.na(all$Outlet_Size)] = "Small"
sum(is.na(all$Outlet_Size))

# Impute Item Visibility where it is '0'

all$Item_Visibility[all$Item_Visibility==0] = NA

all = transform(all,Item_Visibility = na.aggregate(Item_Visibility, by=Item_Identifier))

check = all %>% arrange(Item_Identifier) %>% 
             filter(Item_Identifier %in% c("FDX07","FDP36","NCD19")) %>% 
             select(Item_Identifier, Item_Visibility)


# Create a new variable to get the Item_Identifier first 2 chars
Item_Category = data.frame(Item=c("FD","DR","NC"), Item_txt=c("Food","Drinks","Non-Consumables"))
all$Item_Type_Combined = lookup(substr(all$Item_Identifier, 1,2), Item_Category)
table(all$Item_Type_Combined)


# Create variable for years of operation
all$Outlet_Years = 2013 - all$Outlet_Establishment_Year
all$Outlet_Establishment_Year = NULL
table(all$Outlet_Years)


# Create a new variable to calculate ratio of visibility of an item in a store compared to other stores

all = all %>% mutate(Avg_Visibility = ave(Item_Visibility, by=Item_Identifier, FUN=function(x) mean(x))) %>%
                mutate(Visibility_Index = Item_Visibility / Avg_Visibility) %>%
                select(-Avg_Visibility)

check = all %>% arrange(Item_Identifier) %>%
                 select(Item_Identifier, Outlet_Identifier, Item_Visibility, Visibility_Index)

# Create a new category for the Item Fat content for non-edible items

table(all$Item_Fat_Content, all$Item_Type_Combined)
all$Item_Fat_Content[all$Item_Type_Combined=="Non-Consumables"] = "Non-Edible"

sum(is.na(all))

# One-hot encoding
#-----------------

# Remove the Item_Identifier and Source columns (will add it back later)

all_dummy = all
all_dummy$Item_Identifier = NULL
all_dummy$Source = NULL

dmy = dummyVars("~.",data=all_dummy, fullRank = TRUE)

all_final = data.frame(predict(dmy, newdata=all_dummy))
sum(is.na(all_final))

# Add back those two columns
all_final$Item_Identifier = all$Item_Identifier
all_final$Outlet_Identifier = all$Outlet_Identifier
all_final$Source = all$Source

str(all_final)


#============================================================#
#                       MODEL BUILDING                       #
#============================================================#

# Split the final data back into training and test sets

trainSet = all_final %>% filter(Source=="train") %>%
                         select(-Source, -Item_Identifier, -Outlet_Identifier)
testSet  = all_final %>% filter(Source=="test") %>%
                         select(-Source)


# Feature Selection using RFE

control = rfeControl(functions=rfFuncs,
                     method="repeatedcv",
                     repeats=3,
                     verbose=FALSE)

outvar = "Item_Outlet_Sales"
predvars = names(trainSet)[!names(trainSet) %in% output]

Best_Vars = rfe(trainSet[,predvars], trainSet[,outvar], rfeControl=control)




# Which models are available and what are their tunable parameters
names(getModelInfo())
modelLookup(model="lm")


# Function to create final output file for submission
Write_File = function(output, outfile)
{
  df = data.frame(Item_Identifier = testSet$Item_Identifier, 
                  Outlet_Identifier = testSet$Outlet_Identifier,
                  Item_Outlet_Sales = output)
  write.csv(df, paste0("C:/Users/amit.r/Desktop/Big Mart Sales/",outfile,".csv"), row.names = FALSE)
}

# Set the cross-validation parameters
trainCV = trainControl(method="repeatedcv",
                       number=3,
                       repeats=3)

outvar = "Item_Outlet_Sales"
predvars = names(trainSet)[!names(trainSet) %in% outvar]


# Create OLS model
#-----------------

mod_OLS = train(trainSet[,predvars],trainSet[,outvar],
                method="lm",
                trControl = trainCV)
summary(mod_OLS)

pred_OLS_train = predict(mod_OLS, trainSet)
RMSE(pred_OLS_train, trainSet$Item_Outlet_Sales)

pred_OLS_test = predict(mod_OLS, testSet)
Write_File(pred_OLS_test, "Submit_OLS")




# Create Random Forest model
#---------------------------

mod_RF = train(trainSet[,predvars],trainSet[,outvar],
                method="rf",
                trControl = trainCV,
                importance=TRUE)

varImp(mod_RF)

pred_RF_train = predict(mod_RF, trainSet)
RMSE(pred_RF_train, trainSet$Item_Outlet_Sales)

pred_RF_test = predict(mod_RF, testSet)
Write_File(pred_RF_test, "Submit_RF")



# Create Decision Tree model
#---------------------------

modelLookup(model="rpart")

mod_Tree = train(trainSet[,predvars],trainSet[,outvar],
               method="rpart",
               trControl = trainCV,
               tuneLength=10)

varImp(mod_Tree)

pred_Tree_train = predict(mod_Tree, trainSet)
RMSE(pred_Tree_train, trainSet$Item_Outlet_Sales)

pred_Tree_test = predict(mod_Tree, testSet)
Write_File(pred_Tree_test, "Submit_Tree")


# Create GBM model
#---------------------------

modelLookup(model="gbm")

mod_GBM = train(trainSet[,predvars],trainSet[,outvar],
                 method="gbm",
                 trControl = trainCV,
                 tuneLength=10)

varImp(mod_GBM)

pred_GBM_train = predict(mod_GBM, trainSet)
RMSE(pred_GBM_train, trainSet$Item_Outlet_Sales)

pred_GBM_test = predict(mod_GBM, testSet)
Write_File(pred_GBM_test, "Submit_GBM")

