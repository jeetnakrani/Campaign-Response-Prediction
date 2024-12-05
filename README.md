# Campaign Response Prediction
---
## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Modeling](#modeling)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [License](#license)
---
## Overview
This project focuses on predicting the response of customers to a marketing campaign. The goal is to classify whether a customer is likely to respond or not based on their demographic and financial information.
---
## Installation
To run this project, you'll need to install the necessary dependencies. You can do so by running the following command:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.6 or higher installed.
---
## Dataset
The dataset used in this project contains 56 rows and 9 columns with information about customers and whether they responded to a campaign. The features include:
- `customer_id`: Unique identifier for each customer.
- `age`: Age of the customer.
- `gender`: Gender of the customer.
- `annual_income`: Annual income of the customer.
- `credit_score`: Credit score of the customer.
- `employed`: Employment status of the customer.
- `marital_status`: Marital status of the customer.
- `no_of_children`: Number of children the customer has.
- `responded`: Target variable, indicating whether the customer responded to the campaign.
---
## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the distribution of numerical and categorical features, as well as their relationship with the target variable.

Key Insights:
- The `age` feature is right-skewed.
- No significant outliers were found in the numerical features.
- Features like `age`, `annual_income`, and `credit_score` are positively correlated with the likelihood of responding.
---
## Feature Engineering
New features were created to enhance model performance:
- `family_size`: Total number of family members.
- `years_to_retirement`: Estimated number of years until the customer reaches retirement age (60 years).
---
## Modeling
Various machine learning models were applied to predict campaign responses:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- AdaBoost Classifier
- K-Nearest Neighbors Classifier

GridSearchCV was used to optimize hyperparameters for each model.
---
## Results
The models were evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     |
| Decision Tree       | 1.00     | 1.00      | 1.00   | 1.00     |
| Logistic Regression | 1.00     | 1.00      | 1.00   | 1.00     |
| XGBoost             | 1.00     | 1.00      | 1.00   | 1.00     |
| AdaBoost            | 0.94     | 0.94      | 0.94   | 0.94     |
| K-Nearest Neighbors | 1.00     | 1.00      | 1.00   | 1.00     |

All models except AdaBoost achieved perfect accuracy, precision, recall, and F1 scores.
---
## Conclusion
The models performed exceptionally well, with most achieving 100% accuracy and other key metrics. The Random Forest, Decision Tree, Logistic Regression, XGBoost, and K-Nearest Neighbors models showed excellent results. AdaBoost performed slightly lower but is still a strong model for this task.

While the high accuracy scores indicate strong model performance, the possibility of overfitting exists, especially given the small size of the dataset.
You can add this to your conclusion section for a more detailed explanation on overfitting:

---

**Overfitting?**

Yes, the model performance showing 100% accuracy across all evaluation metrics (accuracy, precision, recall, F1 score) could indicate overfitting, especially if the data used for training and testing is not sufficiently varied or the model is overly complex for the dataset.

### Possible Explanations for Overfitting:

1. **Perfect Accuracy**: Achieving 100% accuracy may indicate that the model has memorized the training data, resulting in poor generalization to unseen data. This can occur when the model is too complex for the size and diversity of the dataset.
   
2. **Small or Simple Dataset**: If the dataset is small or not diverse enough, the model might show high performance during training and testing but fail in real-world applications where data is more varied.

3. **Model Complexity**: Complex models like Random Forest or Decision Trees can overfit if hyperparameters (e.g., number of estimators or tree depth) are not well-tuned, leading to overfitting despite high accuracy on both training and test data.

### How to Detect Overfitting:

- **Cross-Validation**: Use k-fold cross-validation to evaluate the model on multiple data splits. Significant accuracy fluctuations between the training and validation sets suggest overfitting.
  
- **Validation Set Performance**: Evaluate the model on a validation set or new test data that was not used for training. A drop in accuracy indicates overfitting.
  
- **Learning Curves**: Plot training vs. validation accuracy. If training accuracy increases while validation accuracy stagnates or decreases, it suggests overfitting.

### Addressing Overfitting:

- **Regularization**: Apply L1 (Lasso) or L2 (Ridge) regularization to reduce model complexity and prevent overfitting.
  
- **Reduce Model Complexity**: Tune hyperparameters like tree depth or number of estimators for tree-based models to prevent overfitting.
  
- **Cross-Validation**: Use cross-validation to ensure the model generalizes well.
  
- **More Data**: Increasing the size of the training data helps the model generalize better.

- **Ensemble Methods**: Techniques like bagging, boosting, or dropout can help reduce overfitting by combining multiple models.

If the model continues to perform well on new, unseen data, it may not be overfitting. To validate this, use the methods mentioned above to ensure robustness.

---
## License
This project is licensed under the MIT License - see the LICENSE file for details.
