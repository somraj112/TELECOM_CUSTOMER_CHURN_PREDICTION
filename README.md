# Project 5: Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

---

## Project Overview

This project presents the design and implementation of an **AI-driven customer analytics system** that predicts telecom customer churn and evolves toward an intelligent, agent-based retention strategist.

Customer churn significantly impacts business revenue in subscription-based industries. This system leverages classical machine learning techniques to:

* Predict churn probability using historical behavioral data
* Identify key churn drivers
* Provide structured retention strategies based on risk levels

The project is structured into two milestones:

* **Milestone 1:** Classical machine learning techniques applied to historical telecom customer data to predict churn risk and evaluate model performance.
* **Milestone 2:** Extension into an agentic AI system capable of reasoning about churn risk, retrieving best practices using RAG (Retrieval-Augmented Generation), and generating structured intervention plans.

---

## Constraints & Requirements

* **Team Size:** 3–4 Students
* **API Budget:** Free Tier Only (Open-source models / Free APIs)
* **Framework:** LangGraph (Recommended for Milestone 2)
* **Hosting:** Mandatory (Streamlit Cloud / Hugging Face Spaces / Render)

---

## Technology Stack

| Component                | Technology                                        |
| :----------------------- | :------------------------------------------------ |
| **ML Models (M1)**       | Logistic Regression, Decision Tree (Scikit-Learn) |
| **Data Processing**      | Pandas, NumPy, StandardScaler                     |
| **Evaluation Metrics**   | Accuracy, Precision, Recall, F1-Score, ROC-AUC    |
| **UI Framework**         | Streamlit                                         |
| **Agent Framework (M2)** | LangGraph (Planned)                               |
| **Vector Database (M2)** | Chroma / FAISS (Planned)                          |
| **LLMs (M2)**            | Open-source / Free-tier APIs (Planned)            |

---

## Dataset Description

* **Dataset:** Telco Customer Churn Dataset
* **Records:** 7,032 customers
* **Features:** 30 after encoding
* **Target Variable:** `Churn` (Yes/No → 1/0)

### Key Features:

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Internet Service
* Payment Method
* Senior Citizen Status
* Service Subscriptions

---

## Milestones & Deliverables

---

### Milestone 1: ML-Based Churn Prediction (Mid-Sem)

### Objective

Identify customers at risk using historical behavioral data through a classical machine learning pipeline **without LLMs**.

---

### System Architecture (M1)

```
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
One-Hot Encoding
     ↓
Feature Scaling (StandardScaler)
     ↓
Train-Test Split (80/20)
     ↓
Model Training
     ↓
Evaluation & UI Deployment (Streamlit)
```

---

### Model Implementation

Two classification models were implemented and evaluated:

#### 1. Logistic Regression (Final Selected Model)

* Accuracy: **78.7%**
* Better recall & ROC-AUC compared to Decision Tree
* Provides probability outputs for churn risk scoring

#### 2. Decision Tree

* Accuracy: **72.4%**
* Lower generalization performance

Logistic Regression was selected as the final deployment model due to superior overall performance.

---

### Model Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Special focus was placed on **recall for churn class**, since false negatives represent lost customers.

---

### Working Application (Streamlit UI)

The system includes an interactive Streamlit interface that allows:

* Manual customer input
* Real-time churn probability prediction
* Risk categorization:

  * Low Risk
  * Medium Risk
  * High Risk
* Structured retention recommendations

The application loads:

* Trained Logistic Regression model (`.pkl`)
* Feature scaler
* Feature column alignment

---

## Milestone 2: Agentic AI Retention Assistant (End-Sem)

### Objective

Extend the system into an agentic AI strategist that:

* Reasons about churn risk
* Retrieves retention best practices using RAG
* Generates structured intervention plans
* Operates as a multi-state workflow using LangGraph

---

### Planned Agent Workflow

```
Input Customer Profile
        ↓
Risk Assessment Node
        ↓
Knowledge Retrieval (RAG)
        ↓
Strategy Planning Node
        ↓
Structured Retention Report Generation
```

---

### Key Deliverables (M2)

* Publicly deployed application
* Agent workflow documentation (States & Nodes)
* Structured retention report generation
* Complete GitHub repository
* 5-minute demo video

---

## Retention Strategy Logic (Current Rule-Based Prototype)

For Milestone 1, retention suggestions are rule-driven:

* High churn probability (>70%)

  * Offer discounted long-term contract
  * Provide loyalty benefits
  * Assign customer success representative
* Medium risk (40–70%)

  * Personalized promotional offers
  * Engagement follow-ups
* Low risk (<40%)

  * Standard monitoring

This will later evolve into an intelligent agent-based reasoning system.

---

## Project Structure

```
TELECOM_CUSTOMER_CHURN_PREDICTION/
│
├── app.py
├── churn.ipynb
├── logistic_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── telco_dataset.csv
├── requirements.txt
├── README.md
├── .gitignore
```

---

## Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/somraj112/TELECOM_CUSTOMER_CHURN_PREDICTION.git 
cd TELECOM_CUSTOMER_CHURN_PREDICTION
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run Application

```
streamlit run app.py
```

---

## Deployment

The application will be publicly hosted using:

* Streamlit Cloud (Planned)
* Hugging Face Spaces (Optional)
* Render (Optional)

---

## Evaluation Criteria

| Phase       | Weight | Criteria                                                                        |
| :---------- | :----- | :------------------------------------------------------------------------------ |
| **Mid-Sem** | 25%    | ML technique application, Feature Engineering, UI Usability, Evaluation Metrics |
| **End-Sem** | 30%    | Reasoning quality, RAG implementation, Output clarity, Deployment success       |

---

## Future Improvements

* Hyperparameter tuning
* Feature importance visualization dashboard
* SHAP explainability integration
* Real-time business dashboard
* Agentic multi-step reasoning workflow (LangGraph)

---
