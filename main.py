import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ['GROQ_API_KEY'])


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


xgboost = load_model("xgb_model.pkl")
naive_bayes = load_model("nb_model.pkl")
random_forest = load_model("rc_file.pkl")
decision_tree = load_model("dt_model.pkl")
svm = load_model("svm_model.pkl")
knn = load_model("knn_model.pkl")
voting_classifier = load_model("voting_clf.pkl")
xgboost_SMOTE = load_model("xgb-SMOTE.pkl")
xgboost_featureEngineering = load_model("xgb-featureEngineering_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member, est_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': est_salary,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_prediction(input_df):
    probabilities = {
        'XGBoost': xgboost.predict_proba(input_df)[0][1],
        'Random Forest': random_forest.predict_proba(input_df.values)[0][1],
        'K-Nearest Neighbors': knn.predict_proba(input_df.values)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} chance of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    return avg_probability


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that the probability of a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  A customer is at risk of churning only if their predicted churning probability is over 40%. Given predicted churning probability, determine whether they are at risk of churning and generate an explanation (3 paragraphs, max 350 words, third person view).
  Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 11 most important features for predicting churn:
            Feature | Importance
    ----------------------------------
      NumOfProducts |  0.330930
     IsActiveMember |  0.195791
                Age |  0.109685
  Geography_Germany |  0.081833
            Balance |  0.054735
    Geography_Spain |  0.044963
        Gender_Male |  0.043983
        CreditScore |  0.036846
    EstimatedSalary |  0.035971
             Tenure |  0.033146
          HasCrCard |  0.032117

    {pd.set_option('display.max_columns', None)}
    
    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}
    
    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    Don't mention the probability of churning, or the machine learning model, no opening like "If the customer with surname Hargrave has over a 40% risk of churning" or "let's analyze why they might be at risk of churning or not.", just explain the prediction.
  """
    print("\n\nPROMPT:\n", prompt)

    raw_response = client.chat.completions.create(
        model="llama3-groq-8b-8192-tool-use-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }])
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS bank. You are reponsible for ensuring customers stay with the bank and are incentivized with various offers.
    
        You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.
        
        Here is the customer's information:
        {input_dict}
        
        Here is some explanation as to why the customer might be at risk of churning:
        {explanation}
        
        Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, and offering them incentives so that they become more loyal to the bank.
        
        Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model, or importance value, or any detail statistics, to the customer.
        """

    raw_response = client.chat.completions.create(
        model="llama3-groq-8b-8192-tool-use-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }])
    print("\n\nEMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv('churn.csv')
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    print("-----------------------------------------------------")
    print("\nCustomer ID", selected_customer_id)
    selected_customer_name = selected_customer_option.split(" - ")[1]
    print("Customer Name", selected_customer_name)
    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]
    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))
        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=selected_customer['Age'])
        gender = st.selectbox(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == 'Male' else 1)

        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products",
                                       min_value=0,
                                       max_value=20,
                                       value=int(
                                           selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)
    avg_probability = make_prediction(input_df)

    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer_name)

    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer['Surname'])

    st.markdown("---")
    st.subheader("Personalized Email to Customer")
    st.markdown(email)
