import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# LOAD ML COMPONENTS
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# LLM SETUP (GROQ)
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# PREDICTION FUNCTION
def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled_data = scaler.transform(df)

    prob = model.predict_proba(scaled_data)[0][1]

    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "churn_probability": float(prob),
        "risk_level": risk
    }

# LLM REASONING
def analyze_customer(customer_data, prediction):
    prompt = f"""
    You are a telecom retention expert.

    Customer Data:
    {customer_data}

    Prediction:
    {prediction}

    Explain:
    - Why customer may churn
    - Key risk drivers

    Keep it concise in bullet points.
    """
    response = llm.invoke(prompt)
    return response.content

# RAG SETUP
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Customers with low tenure are more likely to churn. Provide onboarding support."),
    Document(page_content="High monthly charges increase churn risk. Offer discounts or better plans."),
    Document(page_content="Month-to-month contracts lead to higher churn. Encourage long-term contracts."),
    Document(page_content="Lack of tech support increases churn. Provide premium support."),
    Document(page_content="Offer loyalty rewards to retain long-term customers."),
]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding)

def retrieve_strategies(query):
    results = vectordb.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

# LANGGRAPH AGENT
from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    customer_data: dict
    prediction: dict
    reasoning: str
    strategies: list
    report: dict

def predict_node(state):
    state["prediction"] = predict_churn(state["customer_data"])
    return state

def reasoning_node(state):
    state["reasoning"] = analyze_customer(
        state["customer_data"],
        state["prediction"]
    )
    return state

def retrieval_node(state):
    state["strategies"] = retrieve_strategies(state["reasoning"])
    return state

def report_node(state):
    state["report"] = {
        "Risk Summary": {
            "Churn Probability": state["prediction"]["churn_probability"],
            "Risk Level": state["prediction"]["risk_level"]
        },
        "AI Reasoning": state["reasoning"],
        "Recommended Actions": state["strategies"]
    }
    return state

graph = StateGraph(AgentState)

graph.add_node("predict", predict_node)
graph.add_node("reason", reasoning_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("generate_report", report_node)

graph.set_entry_point("predict")

graph.add_edge("predict", "reason")
graph.add_edge("reason", "retrieve")
graph.add_edge("retrieve", "generate_report")

graph.add_edge("generate_report", END)

app = graph.compile()

# STREAMLIT UI
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("AI Customer Churn Prediction System")
st.write("Enter customer details to predict churn probability.")

# INPUT FORM
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72)
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])

monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

# BUTTON
if st.button("Predict Churn"):

    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    result = app.invoke({
        "customer_data": input_dict
    })

    report = result["report"]

    st.subheader("Risk Summary")

    prob = report["Risk Summary"]["Churn Probability"]

    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.write(f"**Risk Level:** {report['Risk Summary']['Risk Level']}")
    st.progress(prob)

    st.subheader("AI Reasoning")
    st.write(report["AI Reasoning"])

    st.subheader("Recommended Actions")
    for action in report["Recommended Actions"]:
        st.write(f"• {action}")

    st.subheader("Disclaimer")
    st.write("AI-generated recommendations. Validate before business decisions.")

st.markdown("---")
st.caption("Built using ML + LLM + RAG + LangGraph")