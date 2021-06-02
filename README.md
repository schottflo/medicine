# Project Medicine

## Content
In Project Medicine I co-created an approach to predict medical events and vital parameters of patients based on their medical history in presence of large amounts of missing data (more than 80%). It turned out that the XGBoost Classifier, a tree-based method encodes the NA values automatically and efficiently, relieving the user from the question of how to impute the missing values. Moreover, hand-engineered features improved performance considerably (see script helpers.py). The following features were used to predict the labels:

Time,Age,EtCO2,PTT,BUN,Lactate,Temp,Hgb,HCO3,BaseExcess,RRate,Fibrinogen,Phosphate,WBC,Creatinine,PaCO2,AST,FiO2,Platelets,SaO2,Glucose,ABPm,Magnesium,Potassium,ABPd,Calcium,Alkalinephos,SpO2,Bilirubin_direct,Chloride,Hct,Heartrate,Bilirubin_total,TroponinI,ABPs,pH

Also, there is a column called "pid" that identifies each patient uniquely.

## Requirements
The user needs to save three separate files in the working directory. One of them should contain the feature vectors for the training set, another one the feature vectors for the test set. Third, a separate file for the labels of the training set needs to be created. The predicted labels can be inferred from the scripts.

## Co-authors
David Wissel,
Pascal Kündig
