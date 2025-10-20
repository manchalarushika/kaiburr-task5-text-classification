# kaiburr-task5-text-classification(Data Science Assessment 2025)
This repository contains the submission for Kaiburr Recruitment Assessment 2025 – Task 5 (Data Science Example).
The goal is to perform **text classification** on the official [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) and categorize complaints into multiple product categories using **machine-learning and NLP techniques**.

## Objective

Perform multi-class text classification of customer complaints into the following four categories:

1. **Credit reporting, repair, or other**
2. **Dept Collection**
3. **Consumer Loan**
4. **Mortgage**
---

## Dataset Overview

- **Source:** [Consumer Complaint Database – data.gov](https://catalog.data.gov/dataset/consumer-complaint-database)  
- **Format:** CSV (ZIP) file containing ~300K rows
- **Columns Used:**  
  - `Product` → Target label  
  - `Consumer complaint narrative` → Text data for classification  
- **Sample Used:** 50,000 rows for efficient computation  
- **Language:** English 

<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/d5c56535-faa2-40ec-9057-7484aa831496" />

---

## Workflow Implemented

### 1.Exploratory Data Analysis (EDA) and Feature Engineering
- Dataset loading & preview  
- Product category distribution  
- Complaint length histogram

  <img width="1855" height="900" alt="image" src="https://github.com/user-attachments/assets/e9cdd615-e9ca-4862-85ad-0999782ced01" />

## 2.Text Pre-Processing
- Lower-casing & punctuation removal  
- Stopword removal using **NLTK**  
- Lemmatization using **WordNetLemmatizer**  
- Multiprocessing for parallel text cleaning

<img width="1852" height="892" alt="image" src="https://github.com/user-attachments/assets/ebc2a02f-6b7b-4e45-a40c-fb963e927822" />

### 3.Visualization (EDA Outputs)
- Class distribution plot  
- WordCloud for each category 

### 4.Feature Engineering
- **TF-IDF Vectorizer** (max features = 5000, 1–2 grams)  
- **Label Encoding** for target values
  <img width="1919" height="915" alt="image" src="https://github.com/user-attachments/assets/22d84f37-0ac1-42f2-87a2-c3e336676a95" />


### 5. Model Selection & Training
- **Logistic Regression**  
- **Multinomial Naive Bayes**  
- **Linear SVM (LinearSVC)**  
- Accuracy and runtime comparison
  <img width="1919" height="920" alt="image" src="https://github.com/user-attachments/assets/270e014b-81e5-404e-80b3-73ffc110b5f5" />


### 6.Model Evaluation
- Accuracy, Precision, Recall, F1-Score  
- Classification Report  
- Confusion Matrix heatmap
  <img width="1919" height="926" alt="image" src="https://github.com/user-attachments/assets/e9df84b0-1b3f-44a8-a3c4-ea268da6e913" />


### 7.Prediction Demo
- Custom text input → Predicted Category
  <img width="1918" height="930" alt="image" src="https://github.com/user-attachments/assets/0bb22f78-efef-4d61-b303-2317812efe1f" />


### 8.Model Persistence
- Saved trained model and TF-IDF vectorizer (`best_model.pkl`, `vectorizer.pkl`)

---

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|:--|--:|--:|--:|--:|
| Logistic Regression | **91.10 %** | 0.91 | 0.91 | 0.91 |
| Naive Bayes | 88.70 % | 0.88 | 0.88 | 0.88 |
| Linear SVM | 90.40 % | 0.90 | 0.90 | 0.90 |

**Best Model:** Logistic Regression (≈ 91 % accuracy)  
**Total Training Time:** ≈ 2 minutes (on Colab runtime)
 
##  Setup & Usage

###  Clone the Repository
```bash
git clone https://github.com/manchalarushika/kaiburr-task5-text-classification.git
cd kaiburr-task5-text-classification
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Run the code
```bash
python task5.py
```

 ## Screenshots
 1)Exploratory Data Analysis (EDA) and Feature Engineering
 <img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/88f6b329-07b4-4b83-8217-032baf1b2883" />
 
 2)Text Pre-Processing
 <img width="1918" height="930" alt="image" src="https://github.com/user-attachments/assets/0c8e1946-dd3e-44ef-aa15-31bb7e839752" />

 3)Selection of Multi-Classification Model (Feature Engineering + Split)
 <img width="1919" height="928" alt="image" src="https://github.com/user-attachments/assets/c195f3a5-46fe-4d9b-9721-ab7218ae1f1e" />
 
 4)Comparison of Model Performance
 <img width="1918" height="924" alt="image" src="https://github.com/user-attachments/assets/5ebffe62-e119-48a6-b1e9-7d3718a2c844" />

5)Model Evaluation
<img width="1919" height="922" alt="image" src="https://github.com/user-attachments/assets/99f27767-4b57-41b1-ab15-7209aae14d22" />

6. Prediction
   <img width="1919" height="922" alt="image" src="https://github.com/user-attachments/assets/dd166ad2-fc38-4a1e-9c1a-8464a27d86e3" />






 





