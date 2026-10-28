# Titanic Survival Prediction: Classical ML Model Comparison

A machine learning project that predicts passenger survival on the Titanic and compares five classification algorithms: Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM.

## Problem Statement
Given a passenger's details (class, sex, age, family aboard, fare, port of embarkation), predict whether they survived (`1`) or not (`0`). This is a binary classification problem.

## Dataset
- **Source:** Titanic dataset bundled with Seaborn (`sns.load_dataset('titanic')`), so no manual download is needed.
- **Size:** 891 rows × 15 columns (889 rows after cleaning)
- **Target:** `survived`
- **Features used:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`

## Approach
1. **Cleaning:** dropped redundant or sparse columns (`class`, `who`, `adult_male`, `deck`, `embark_town`, `alive`), filled missing `age` with the mean, and dropped the 2 rows with missing `embarked`.
2. **Encoding:** label-encoded `sex` and `embarked`.
3. **Split:** 80/20 train/test split (`random_state=42`).
4. **Scaling:** `StandardScaler` for KNN, Decision Tree and SVM.
5. **Models:** Logistic Regression, KNN (k=5), Gaussian Naive Bayes, Decision Tree, SVM (RBF kernel).
6. **Evaluation:** accuracy, confusion matrix, precision/recall/F1, plus 5-fold cross-validation for the SVM.

## Results

Test set of 178 passengers:

| Model | Accuracy | Precision (survived) | Recall (survived) | F1 (survived) |
|-------|----------|----------------------|-------------------|---------------|
| Logistic Regression | 0.803 | 0.74 | 0.77 | 0.75 |
| KNN (k=5) | 0.792 | 0.72 | 0.75 | 0.74 |
| Naive Bayes | 0.775 | 0.68 | 0.78 | 0.73 |
| Decision Tree | 0.803 | 0.74 | 0.77 | 0.75 |
| **SVM (RBF)** | **0.815** | **0.75** | **0.78** | **0.77** |

**5-fold cross-validation (SVM):** scores of 0.831, 0.820, 0.815, 0.809, 0.864, with a **mean accuracy of 0.828**.

SVM performed best on the held-out test set, though all five models land within about 4 percentage points of each other. The cross-validated score shows the result holds up across different splits.

## Tech Stack
Python · pandas · NumPy · scikit-learn · seaborn · matplotlib · Jupyter Notebook

## Getting Started
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook ML_Project4_.ipynb
```

## Project Structure
```
├── ML_Project4_.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Author
<Aridaman singh>
