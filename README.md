# Predictive Performance vs Interpretability in Telecom Churn Prediction

End-to-end machine learning project evaluating predictive performance, interpretability, and deployment strategies for telecom customer churn.

MSc Data Analytics dissertation (De Montfort University) comparing how well machine learning models predict telecom customer churn against how easily their decisions can be explained. The main comparison is **XGBoost** against **logistic regression**, with **SHAP** used to open up the models and see what is actually driving each prediction, and a customer-lifetime-value layer on top to turn churn scores into retention decisions.

## What this project asks

The work is built around three research questions:

1. How does an ensemble model (XGBoost) perform on imbalanced churn data when judged by metrics suited to imbalance, rather than raw accuracy?
2. Where does SHAP attribution agree with, and diverge from, the coefficients of a logistic regression model on the same data?
3. How can a churn probability be turned into a retention decision that accounts for customer lifetime value, rather than just targeting the highest-risk customers first?

## Data

This project uses the **Iranian Churn Dataset**, originally published in the UCI Machine Learning Repository and also mirrored on Kaggle. It was randomly collected from an Iranian telecom company's database over a 12-month period and contains 3,150 customer records with no missing values, where each row is one customer. All of the features are aggregated from the customer's first 9 months of activity, while the churn label reflects their status at the end of the 12 months.

Churn is imbalanced: 495 of the 3,150 customers churned, roughly 15.7 percent. That imbalance is the reason the modelling leans on PR-AUC rather than raw accuracy.

The 13 features cover call behaviour and account profile: call failures, number of complaints, subscription length, charge amount, seconds of use, frequency of use, frequency of SMS, distinct called numbers, age group, age, tariff plan, account status, and a precomputed Customer Value score. The target is a binary `Churn` label. The copy in this repository also carries two extra columns, `FN` and `FP`, which are derived during the analysis and are not part of the original dataset.

**Licence and attribution.** The dataset is released under a Creative Commons Attribution 4.0 (CC BY 4.0) licence, so it can be shared and adapted as long as the source is credited. Cite as:

> Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A. (2020). Optimum Profit-Driven Churn Decision Making: Innovative Artificial Neural Networks in Telecom Industry. *Neural Computing and Applications*. UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z

## Approach

- Preprocessing handled separately for each model, with all transformations fit on the training set only to avoid leakage into the test set.
- SMOTE used to address class imbalance in the training data.
- Box-Tidwell test used to check the linearity assumption behind logistic regression.
- Models compared on PR-AUC rather than AUC-ROC, since PR-AUC is the more honest measure when the positive class (churners) is rare.
- SHAP values computed for the XGBoost model and set side by side with the logistic regression coefficients.

## Key findings

Performance on the held-out test set, with all metrics reported for the minority (churn) class:

| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.924 | 0.674 | 0.633 | 0.503 | 0.854 |
| Random Forest | 0.978 | 0.871 | 0.800 | 0.694 | 0.944 |
| **XGBoost** | **0.983** | **0.885** | **0.812** | **0.726** | 0.921 |

- **XGBoost wins, but the real story is precision, not AUC.** The AUC-ROC gain over logistic regression is modest (+6.4 points), because LR already discriminates well on this fairly separable data. The gain that matters is F1 (+17.9 points): logistic regression's precision of 0.50 means about half of flagged customers are false positives, while XGBoost at 0.73 cuts that waste sharply. On a fixed retention budget, that is the difference worth having.
- **Subscription length is genuinely nonlinear.** A Box-Tidwell test confirmed nonlinearity (p = 0.017), so logistic regression's linear log-odds can only fit a shallow slope and ranks the feature near the bottom (10th). SHAP, without that constraint, picks up the protection that accelerates at long tenure and ranks it 6th.
- **SHAP and logistic regression agree more than they disagree.** Across the 11 features the rankings split into 5 in close agreement, 4 with moderate differences, and 2 that diverged substantially. The two divergences are interpretable rather than contradictory: subscription length (above), and complaints, where LR captures the strong local effect of a complaint (odds ratio around 3.0) while SHAP averages it down because over 90 percent of customers never complain. The two methods are answering different questions.
- **Engagement is the dominant churn signal.** The top SHAP features are all engagement measures: frequency of use, call failures, seconds of use, and frequency of SMS. Disengagement, rather than any single account attribute, is the clearest lead indicator.
- **CLV-weighted targeting beats highest-risk-first.** Simulating a fixed retention budget on the test set, ranking customers by churn probability weighted by customer value retained about 1.4x more expected revenue than chasing the highest-risk customers first, because the highest-risk group skews toward low-value customers.

## Repository structure

```
.
├── notebooks/        # analysis and modelling notebooks
├── src/              # any reusable scripts
├── data/             # data or a pointer to the source (see Data section)
├── dissertation.pdf  # full written dissertation
├── requirements.txt  # dependencies
└── README.md
```

## Running the code

The notebook was written and run in Google Colab, so the first cell installs its own dependencies with `!pip install` and reads the data from a Colab path (`/content/Customer Churn (1).csv`). It runs top to bottom with a fixed random seed (42), so a clean sequential run reproduces every result.

**In Colab:**

1. Upload the notebook and the CSV to your Colab session.
2. Check that `DATA_PATH` in the first cell points at your uploaded file. The CSV in this repo is named `Customer_Churn__1_.csv`, so either rename it to match the path or update `DATA_PATH` to the new name.
3. Run all cells.

**Locally:**

```bash
git clone https://github.com/Ibracadabrah/telecom-churn-dissertation.git
cd telecom-churn-dissertation
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook and set `DATA_PATH` to the local CSV (for example `data/Customer_Churn__1_.csv`). Once the dependencies are installed you can comment out the `!pip install` line in the first cell.

Built and tested on Python 3 with scikit-learn 1.6.1, xgboost 3.2.0, imbalanced-learn 0.14.1, shap 0.51.0, statsmodels 0.14.6, pandas 2.2.2, and numpy 2.0.2.

## Author

Ibrahim — MSc Data Analytics, De Montfort University.
