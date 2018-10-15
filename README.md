# Telecom-Churn-Analysis
Customer churn refers to when a customer (player, subscriber, user, etc.) ceases his or
her relationship with a company. Businesses typically treat a customer as churned once a
particular amount of time has elapsed since the customer’s last interaction with the site or
service. The full cost of customer churn includes both lost revenue and the marketing
costs involved with replacing those customers with new ones. Reduction customer churn
is important because cost of acquiring a new customer is higher than retaining an existing
one. Reducing customer churn is a key business goal of every business. This case is
related to telecom industry where particular organizations want to know that for given
certain parameters whether a person will churn or not.
### Exploratory Data Analysis
Our target variable has two categories which include True and False values.
True = Customer will move or churn out.
False = Customer won’t move
We can clearly see that our data is highly imbalanced. The occurrence of false
is higher than True. There are 2850 (85.51% ) customers who churn out and 483 (14.49%)
customers retained.
![customer churn](https://user-images.githubusercontent.com/20225277/46948212-de6c9880-d09a-11e8-8c32-3cb7587ba9fd.png)

###### State Wise Churn Analysis
![statewise](https://user-images.githubusercontent.com/20225277/46948261-065bfc00-d09b-11e8-9837-3f5fdfb42d2d.png)

###### Area Wise Churn Analysis
Most of the churned customers are from 415 area.
![area_wise](https://user-images.githubusercontent.com/20225277/46948273-1bd12600-d09b-11e8-81d6-a263377a496d.png)

###### Churn according to International Plan
Churn rate is more with customer using international plan. As only 323 customer
using International plan and 137 churning out of them.
![planvise](https://user-images.githubusercontent.com/20225277/46948340-5fc42b00-d09b-11e8-9be0-20c26b06ed21.png)

###### Churn according to Voicemail Plan
922 customer using voice mail plan and 80 out of them are churning
![voice mail plan vise](https://user-images.githubusercontent.com/20225277/46948361-74082800-d09b-11e8-9c83-f9a047633dfa.png)

###### Churn according to Customer Care Calls
Churn rate for Customer neither having voicemail plan nor international plan is
9.06%. Churning rate for customer having International plan but don’t have voicemail plan is
3.03% out of 6.93% customers. Churning of customer having both voicemail plan & international plan is 1.08% out of
2.76% 
![customer care calls vise](https://user-images.githubusercontent.com/20225277/46948289-2b506f00-d09b-11e8-8e39-1826c2a5eff8.png)

###### Collinear Plot
‘Total day minutes’ and ‘total day charges’ are highly correlated
‘Total eve minutes’ and ‘total eve charges’ are highly correlated
‘Total night minutes’ and ‘total night charges’ are highly correlated
‘Total intl minutes’ and ‘total intl charges’ are highly correlated
![correlation plot_py](https://user-images.githubusercontent.com/20225277/46948308-3dcaa880-d09b-11e8-8504-c9240fb55b92.png)

##### Distribution of variables
Most of thevariables are normally distributed.
![nr1](https://user-images.githubusercontent.com/20225277/46948317-49b66a80-d09b-11e8-8430-92757348c078.png)

### SMOTE Oversampling in Python
SMOTE synthesize new minority instances between existing real minority instances.
Imagine that SMOTE Draw lines between existing minority instances. SMOTE then imagine
new synthetic minority instance somewhere on that lines. Like it will generate the synthetics
of two real minority cases or data points. Applying synthetic minority oversampling technique
to overcome the challenge of imbalance dataset as having an imbalance dataset will have
negative impact over machine learning model predictions.
In python we use SMOTE
###### Before :-
False = 1895 True = 338
###### After Smote
False = 1895 True = 1895
### ROSE Oversampling in R:-
IN R we have used ROSE sampling technique. Which is similar to SMOTE, It also
generating the synthetic data points and also it will under sample some random points from
majority class.
###### Before :-
False = 1881 True = 319
###### After Smote
False = 1101 True = 1019

### Model Evaluation
1. Random Forest
2. Logistic Regression
3. K- Nearest Neighbors
4. Naïve Bayes

### Model Selection
Random forest has the best results for our problem. Random Forest has the best accuracy
and lowest false negative rate and also lowest false positive rate.Hence we’ll choose
Random Forest.

