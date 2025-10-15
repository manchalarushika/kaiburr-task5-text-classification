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

##Dataset Overview

- **Source:** [Consumer Complaint Database – data.gov](https://catalog.data.gov/dataset/consumer-complaint-database)  
- **Format:** CSV (ZIP) file containing ~300K rows
- **Columns Used:**  
  - `Product` → Target label  
  - `Consumer complaint narrative` → Text data for classification  
- **Sample Used:** 50,000 rows for efficient computation  
- **Language:** English 

---

## Workflow Implemented

###1.Exploratory Data Analysis (EDA)
- Dataset loading & preview  
- Product category distribution  
- Complaint length histogram

## 2.Text Pre-Processing
- Lower-casing & punctuation removal  
- Stopword removal using **NLTK**  
- Lemmatization using **WordNetLemmatizer**  
- Multiprocessing for parallel text cleaning  

### 3.Visualization (EDA Outputs)
- Class distribution plot  
- WordCloud for each category 

### 4.Feature Engineering
- **TF-IDF Vectorizer** (max features = 5000, 1–2 grams)  
- **Label Encoding** for target values 

### 5. Model Selection & Training
- **Logistic Regression**  
- **Multinomial Naive Bayes**  
- **Linear SVM (LinearSVC)**  
- Accuracy and runtime comparison 

### 6.Model Evaluation
- Accuracy, Precision, Recall, F1-Score  
- Classification Report  
- Confusion Matrix heatmap

### 7.Prediction Demo
- Custom text input → Predicted Category  

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
  -Dataset load confirmation
 <img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/d5c56535-faa2-40ec-9057-7484aa831496" />
 
  -EDA visualizations
 <img width="1855" height="900" alt="image" src="https://github.com/user-attachments/assets/e9cdd615-e9ca-4862-85ad-0999782ced01" />

 - Model accuracy bar chart
   <img width="1860" height="897" alt="image" src="https://github.com/user-attachments/assets/e1a8d0f4-53e5-40ec-a2c7-c50663ad89c6" />

 - Confusion matrix & Classification report
   <img width="1862" height="895" alt="image" src="https://github.com/user-attachments/assets/5f8f3863-285b-43d0-84eb-e91683a05b3e" />

 - Sample prediction output
   <img width="1919" height="920" alt="image" src="https://github.com/user-attachments/assets/07d6f6d2-1e6e-41ef-ac41-6defdf54d58d" />





