# Titanic-Dead-Or-Alive

# Introduction

This project analyzes the famous Titanic dataset to understand factors affecting survival rates and build machine learning models to predict passenger survival. This project includes:
- The cleaning of the dataset
- Visualisations
- Feature engineering
- Statistical tests
- Predictive modelling using Logistic Regression, K-Nearest Neighbours (KNN) and Random Forest

# Dataset

The dataset used contains information about each passenger onboard the titanic and their survival status. Lived(1), Died(0). Each passengers has several features such as:
- Pclass (passenger class)
- Name
- Sex
- Age
- SibSp (Number of siblings and spouses onboard)
- Parch (Number of parents and children onboard)
- Fare
- Cabin
- Embarked (Where they boarded the titanic)

The dataset can be found at - 

# Methodology

# 1. Dataset Cleaning and Preprocessing

The dataset was cleaned by:
- Removing any influential outliers
- Creating visualisations to see the spread of certain features
- Handling missing data using mean/median values as well as an iterative imputer
- Mapping one categorical column into several numerical columns ready for the machine learning models
- Created new features such as family_size, is_alone, is_child, is_senior, cabin_deck and title
- Prior to each machine learning model features were scaled and SMOTE oversampling was used to balance the outcome variable.

# 2. Statistical Tests

To explore the relationship between some of the features, these tests were conducted:
- One sample T-Test
  Checked if the mean fare price for people who embarked at C differed from the hypothesized population mean of the Titanic.
- Two sample T-Test
  Compared the mean ages for men against the mean ages for women to see who had significantly greater ages on the Titanic.
- Two sample T-Test
  Compared the mean fare for men against the mean fare for women to see who had significantly higher fare prices.
- Chi-squared Test
  Checked to see if there is an association between survival status and gender.
- Chi-squared Test
  Checked to see if there is an association between cheap/expensive fares and survival status.

Key Findings
- The people who embarked at C paid significantly greater than the rest of the people who boarded the Titanic.
- The mean age for men was significantly greater than the mean age of women.
- The mean male fare was significantly lower than the mean female fare.
- There is an association with survival status and gender. More women survived.
- There is an association between survival status and cheap/expensive tickets. People with cheaper tickets tended not to survive.

# 3. Models

The following models are trained and evaluated:
- Logistic Regression: A linear model for classification
- K-Nearest Neighbours (KNN): A distance based algorithm for classification
- Random forest: An ensemble method using decision trees for classification

# 4. Evaluation Metrics

The models were evaluated using:
- Accuracy: Fraction of correctly classified samples.
- Precision: Fraction of true positives among predicted positives.
- Recall: Fraction of true positives among actual positives.
- F1-Score: Harmonic mean of Precision and Recall.
- Confusion matrix - Showing all points classified as either true negative, false positive, false negative and true positive.
- ROC-AUC: Area under the Receiver Operating Characteristic curve.

# Installation

- Clone the repostiory using:
  git clone https://github.com/AdamBartlett7/Titanic-Dead-Or-Alive.git
- Navigate to the correct directory using:
  cd Titanic-Dead-Or-Alive
- To create your own virtual environment with the necessary python libraries use:
  conda env create -f environment.yml
- Open and run the files.

# Usage
- Run Titanic_cleaning.ipynb in sequential order to see the cleaning of the dataset, some visualisations on the features and feature engineering.
- Run titanic_stat_tests.ipynb in sequential order to see the statistical tests and the resulting p-values.
- Run Titanic_log_reg.ipynb in sequential order to see how logistic regression model was created and performed also how the hyperparameters were tuned.
- Run Titanic_knn.ipynb in sequential order to see how KNN model was created and performed also how the number for k was optimised.
- Run Titanic_rf.ipynb in sequential order to see how random forest model was created and performed also how the hyperparameters were tuned.

# Results

 | Model               |   Accuracy  |   Precision  |   Recall   |     F1    |    ROC-AUC   | 
 | ------------------- | ----------- | ------------ | ---------- | --------- | ------------ |
 | Logistic Regression |    93.6%    |     52.2%    |    97.8%   |   68.1%   |     98.3%    |
 | KNN                 |    97.7%    |     75.1%    |    100%    |   85.8%   |     99.9%    |
 | Random Forest       |    99.9%    |     100%     |    98.5%   |   99.3%   |     99.9%    |

# ROC Curve
- To evaluate the performance of the models, I plotted the ROC curves for all three models. The ROC curve illustrates the trade-off between True Positive Rate and False 
  Positive Rate, helping us assess the discriminatory power of each model. A higher Area Under the Curve (AUC) indicates better model performance.
  
- Logistic Regression ROC Curve

- KNN ROC Curve

- Random Forest ROC Curve

# Confusion Matrix

   | Log Reg   | Predicted Not Fraud | Predicted Fraud |
   | --------- | ------------------- | --------------- |
   | Not Fraud |         1670        |       120       |
   | Fraud     |           3         |       131       |

   | Log Reg   | Predicted Not Fraud | Predicted Fraud |
   | --------- | ------------------- | --------------- |
   | Not Fraud |         1670        |       120       |
   | Fraud     |           3         |       131       |

   | Log Reg   | Predicted Not Fraud | Predicted Fraud |
   | --------- | ------------------- | --------------- |
   | Not Fraud |         1670        |       120       |
   | Fraud     |           3         |       131       |


# Feature Coefficient & Feature Importance


# Learning Curve

- Logistic Regression Learning Curve

- KNN Learning Curve

- Random Forest Learning Curve

# License
- This project is licensed under the MIT License.

  
  

