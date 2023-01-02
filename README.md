# Credit_Risk_Analysis

Apply machine learning to solve a real-world challenge: credit card risk.

## Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In this project, we are going to to employ different techniques to train and evaluate models with unbalanced classes using imbalanced-learn and scikit-learn libraries.

We are going to use the credit card credit dataset from LendingClub, a peer-to-peer lending services company to:

   * Oversample the data using the RandomOverSampler algorithms.
   * Oversample the data using the SMOTE algorithms.
   * Undersample the data using the ClusterCentroids algorithm.
   * Use a combinatorial approach of oversampling and undersampling using the SMOTEENN algorithm.
   * Compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

### Aim
The aim of this project is to evaluate the performance of 6 machine learning models in order to find the most suitable one to predict credit risk.

## Analysis of Data

### First Model: Naive Random Oversampling Model





 * The total number of low risk (17104) is so high compared to the total number of high risk (101).
 * Model accuracy is 0.66 or 66%, since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk is extremely low (0.01) which reflect the low number of True Positive high risk (75) corresponding to the total number of positive low and        high risk of our model gets (TP+FP = 75+7016 = 7091). However, the sensitivity of detecting true high risks is 0.74 (TP/TP+FP = 75/101=0.74), which means 26 cases of    101 high risk were not detected.
 * Sensitivity for low risk is 0.59 which means that 10088 are actually low risk while 17104 cases were detected low risk.
 
 
### Second Model: SMOTE Oversampling Model 
 
 
 
 
 * The total number of low risk (17104) is so high compared to the total number of high risk (101).
 * Model accuracy is 0.65 or 65%,which is slightly less than that of Naive Random Oversampling Model.
 * Since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk is still extremely low (0.01) which reflect the low number of True Positive high risk (64) corresponding to the total number of positive low    and high risk of our model gets (TP+FP = 64+5523 = 5587). However, the sensitivity of detecting true high risks has decreased to 0.63 (TP/TP+FP = 64/101=0.63),        which means 37 cases of 101 high risk were not detected.
 * Sensitivity for low risk has increased to 0.68, which means that 5523 cases are actually low risk while they have been detected as high risk.
 
 ### Third Model: Undersampling Model
 
 
 
 
 
 * Model accuracy in the Undersampling model has decreased to 0.53 or 53%,which is less than both Oversampling Models.
 * Since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk is still extremely low (0.01) which reflects the low number of True Positive high risk (68) corresponding to the total number of positive      low and high risk of our model gets (TP+FP = 68+ 10500 = 5587). However, the sensitivity of detecting true high risks is 0.67 (TP/TP+FP = 68/101=0.67), which means    33 cases of 101 high risk were not detected in this model.
 * Sensitivity for low risk has decreased to 0.39, which means that 10500 cases are actually low risk while they have been detected as high risk.
 
 ### Fourth Model: Combination (Over and Under) Sampling
 
 
 * Model accuracy in the Combination Sampling model is to 0.64 or 64%, which is so close to that of both Oversampling Models.
 * Since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk is still extremely low (0.01) which reflects the low number of True Positive high risk (75) corresponding to the total number of positive      low and high risk of our model gets (TP+FP = 75+ 7570 = 7645). However, the sensitivity of detecting true high risks is 0.74 (TP/TP+FP = 75/101=0.74), which means      26 cases of 101 high risk were not detected in this model.
 * Sensitivity for low risk is 0.56, which means that 7570 cases are actually low risk while they have been detected as high risk.
 
 
 ### Fifth Model: Balanced Random Forest Classifier



 * Model accuracy in the balanced forest classifier has increased to 0.79 or 79%, which is higher than all previous models.
 * Since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk is slightly bigger than previous models but it's still low (0.04) which reflects the low number of True Positive high risk (72)                corresponding to the total number of positive low and high risk of our model gets (TP+FP = 72+1979 = 2051). Moreover, the sensitivity of detecting true high risks      is 0.71 (TP/TP+FP = 72/101=0.71), which means 29 cases of 101 high risk were not detected in this model.
 * Sensitivity for low risk has increased to 0.88, which means that 1979 cases are actually low risk while they have been detected as high risk. This number is less      than what we got in previous models.
 
 
 ### Sixth Model: Easy Ensemble AdaBoost Classifier
 
 
 
 
 * Model accuracy in the easy ensemble AdaBoost has increased to 0.93 or 93%, which is the highest value of all models.
 * Since the data in unbalanced we need to look into precision and sensitivity.
 * Precision for high risk has increased to 0.09. However, this low value reflects the low number of True Positive high risk (93) corresponding to the total number of    positive low and high risk of our model gets (TP+FP = 93+970=1063). Moreover, the sensitivity of detecting true high risks has increased to 0.92 (TP/TP+FP =            93/101=0.92), which means only 8 cases of 101 high risk were not detected in this model.
 * Sensitivity for low risk has also increased to 0.94, which means that only 970 cases are actually low risk while they have been detected as high risk. This number      is less than what we got in all previous models.
 
## Conclusion
Based on all the above, I would recommend using the Easy Ensemble AdaBoost Classifier algorithm because:

 * It has the highest accuracy score (93%)
 * It has the highest sensitivity of detecting true high risks (0.92); Which means only 8 cases with high risks were not detected using this model,
 * It has the highest sensitivity for low risk (0.94); Which means only 970 cases of 17104 were detected as high risk while in fact they are low risk.
