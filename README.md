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

Extend the system beyond prediction into an **intelligent AI-powered decision-making system** that not only predicts churn but also:

- Explains *why* a customer is likely to churn  
- Retrieves domain knowledge using RAG  
- Generates actionable retention strategies  
- Operates through a structured multi-step workflow using LangGraph  

---

### System Architecture (M2)

```

User Input (Customer Data)
↓
ML Prediction (Logistic Regression)
↓
LLM Reasoning (Groq - LLaMA 3)
↓
Knowledge Retrieval (RAG - ChromaDB)
↓
Strategy Generation
↓
Structured Report Output (Streamlit UI)

```

---

### Agent Workflow (LangGraph)

The system is implemented as a **state-based workflow graph** using LangGraph:

```

predict → reason → retrieve → generate_report → END

```

#### Nodes Description:

- **predict_node**
  - Uses trained ML model to calculate churn probability & risk level  

- **reasoning_node**
  - Uses LLM (Groq - LLaMA 3) to explain churn drivers  

- **retrieval_node**
  - Uses RAG (ChromaDB + Embeddings) to fetch relevant retention strategies  

- **report_node**
  - Combines all outputs into a structured business-ready report  

---

### Key Features Implemented (M2)

- AI-powered churn reasoning (LLM-based)  
- Retrieval-Augmented Generation (RAG) for best practices  
- Multi-step agent workflow using LangGraph  
- Structured report generation (Risk + Reasoning + Actions)  
- Real-time inference through Streamlit UI  
- Deployed live application (Streamlit Cloud)  

---

### Output Structure

The system generates a structured response:

#### 1. Risk Summary
- Churn Probability (%)  
- Risk Level (Low / Medium / High)  

#### 2. AI Reasoning
- Why the customer may churn  
- Key behavioral risk drivers  

#### 3. Recommended Actions
- Data-driven retention strategies retrieved via RAG  

---

### Example Output

```

Churn Probability: 75.74%
Risk Level: HIGH

AI Reasoning:

* New customer with low tenure
* Month-to-month contract indicates low commitment
* High monthly charges

Recommended Actions:

* Offer long-term discounted plans
* Provide onboarding support
* Introduce loyalty rewards

```

---

### Technologies Used (M2 Implementation)

| Component            | Technology Used                    |
|---------------------|----------------------------------|
| LLM                 | Groq (LLaMA 3 70B)               |
| Agent Framework     | LangGraph                        |
| RAG Pipeline        | ChromaDB + HuggingFace Embeddings|
| Orchestration       | Python                           |
| Deployment          | Streamlit Cloud                  |

---

### Improvements Over Milestone 1

| Feature              | Milestone 1 | Milestone 2 |
|---------------------|------------|------------|
| Churn Prediction     | ✅          | ✅          |
| Explainability       | ❌          | ✅ (LLM)    |
| Strategy Suggestions | Rule-based | AI + RAG    |
| System Intelligence  | Static     | Dynamic     |
| Workflow Automation  | ❌          | ✅ (LangGraph) |

---

### Challenges Faced

- Dependency conflicts during deployment (Python version compatibility)  
- LangGraph workflow errors (dead-end nodes, incorrect edges)  
- Streamlit Cloud limitations for heavy libraries  
- Managing API keys securely (Groq API)  

---

### Learnings

- Real-world AI systems require **orchestration, not just models**  
- LLMs enhance explainability but need structured pipelines  
- RAG improves reliability over pure LLM responses  
- Deployment is often more complex than development  

---