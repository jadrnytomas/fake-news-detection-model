# Fake News Classification Using Word2Vec and Machine Learning

This project focuses on the detection and classification of fake news using advanced Natural Language Processing (NLP) techniques and machine learning models. The project was designed for my Machine Learning course.
Dataset for the project was the ISOT Fake News Dataset from Kaggle.
Link: https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset

## 📌 Main Project Objectives
1. **Exploratory Data Analysis (EDA)** using the `ydata-profiling` library.
2. **Complete Text Preprocessing** (cleaning, stop-word removal, and lemmatization using the `spaCy` library).
3. **Text Vectorization** by training a custom `Word2Vec` model (Gensim).
4. **Classifier Training and Optimization** (`Logistic Regression`, `Random Forest`, `XGBoost`) using grid search (`GridSearchCV`).
5. **Critical Model Analysis** focusing on detecting systemic bias (the influence of specific words like *"Reuters"*) and analyzing overfitting.

---

## 🛠️ Technologies and Libraries Used
* **Programming Language:** Python 3
* **Data Processing:** `pandas`, `numpy`
* **NLP & Text Processing:** `spaCy` (model `en_core_web_sm`), `nltk`, `gensim` (Word2Vec), `re`
* **Machine Learning:** `scikit-learn` (Logistic Regression, Random Forest, GridSearchCV, metrics), `xgboost`
* **Visualization & Reporting:** `ydata-profiling` (ProfileReport)

---

## 📈 Methodology and Steps

### 1. Data Loading and Exploratory Data Analysis (EDA)
* We loaded the `True.csv` and `Fake.csv` datasets, labeled them with the target variable `is_fake` (0 for true, 1 for fake news), and concatenated them into a single DataFrame.
* We generated an interactive `ydata-profiling` report to deeply understand the data distribution.
* **Analytical question from the assignment:** *How many articles have a category that contains the word "politics" in its name?*
  * **Answer:** A total of **18,113** articles fall under political topics.

### 2. Data Cleaning and Normalization
* We merged the article title (`title`) and the body text (`text`) into a single column `full_text`.
* We removed unnecessary columns (`subject`, `date`) that could introduce noise into the model.
* We normalized the text by converting it to lowercase and removing numbers, special characters, and redundant spaces.
* We performed **lemmatization** and filtered out stop-words using efficient parallel processing with `nlp.pipe` in `spaCy`.

### 3. Vectorization (Word2Vec)
* We split the data into training and test sets (80/20).
* We trained a `Word2Vec` embedding model on the training texts with a vector dimension of 100.
* For each article, we calculated the **averaged Word2Vec vector** (Averaged Word2Vec) to obtain a fixed-size representation suitable for classification algorithms.

### 4. Model Development & Hyperparameter Optimization
In the first phase, we trained baseline models without hyperparameter tuning. Subsequently, we used `GridSearchCV` with 3-fold cross-validation to optimize key parameters:
* **Logistic Regression** (optimized `C` and `solver`)
* **Random Forest** (optimized `n_estimators`, `max_depth`, `min_samples_split`)
* **XGBoost** (optimized `learning_rate`, `max_depth`, `n_estimators`)

---

## 📊 Results and Model Comparison

### Baseline Model Results (F1-Score / Accuracy):
* **Logistic Regression:** Accuracy ~97.05%
* **Random Forest:** Accuracy ~96.69%
* **XGBoost:** Accuracy ~97.71%

### Results After Hyperparameter Optimization (GridSearchCV):
Optimization achieved a small but consistent increase in accuracy and robustness across all models.

| Model | Best Parameters | Overall Accuracy | Precision (Fake News) | F1-Score (Fake News) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `{'C': 10.0, 'solver': 'lbfgs'}` | **97.17%** | 0.9758 | 0.9729 |
| **Random Forest** | `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}` | **96.55%** | 0.9605 | 0.9672 |
| **XGBoost** | `{'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 100}` | **97.85%** | **0.9769** | **0.9795** |

* **Winning Model:** The best results were achieved by the **XGBoost** model with an overall accuracy of **97.85%** and a precision of **97.69%** for detecting Fake News.

---

## 🔍 Critical Analysis & Interpretation (Added Value)

### 1. The Influence of Source Bias (The Word "Reuters")
During data analysis, we noticed that almost all true news articles originate from the Reuters agency and contain "Reuters" in the text. To verify whether the models were simply learning to recognize the brand name "Reuters" instead of actual truthfulness, we removed this word from the dataset and retrained the models:
* The **XGBoost** model saw a slight decrease in accuracy to **97.48%**.
* This result confirms the hypothesis that the model partially relied on the presence of the word "Reuters".
* **Conclusion for real-world application:** To make the model truly deployable in a production environment, it is essential to expand the training set with true articles from other news agencies (e.g., AP, BBC, DW) to prevent overfitting to the publishing style of a single source.

### 2. Overfitting Analysis
By comparing the accuracy on the training and test sets (without the word "Reuters"), we observed the following gaps:
* **Logistic Regression:** Difference of only **0.0024** (highly stable, almost no overfitting).
* **Random Forest:** Difference of **0.0353** (train accuracy 99.96% vs. test accuracy 96.44%).
* **XGBoost:** Difference of **0.0247** (train accuracy 99.96% vs. test accuracy 97.48%).

*Random Forest and XGBoost exhibit minor overfitting, indicating room for further regularization (e.g., limiting max_depth or increasing min_samples_split).*

---

## 🚀 How to Run Locally

**This project was done in Google Colab**
1. Download the dataset from Kaggle (https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)
2. Upload and open `fake_news_detection_model.ipynb` in Google Colab
3. Upload both CSV's (True.csv, Fake.csv) into the environment runtime files
4. Run the Notebook
