import pandas as pd
from math import floor
from datetime import datetime
import nltk
nltk.download('punkt')
import random
from dateutil.relativedelta import relativedelta
import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import openai

def add_user(first_name, last_name, age, phone_no, monthly_salary):
    """
    Add a new user to the customer data CSV file.

    This function reads existing customer data from a CSV file, generates a new 
    customer ID for the new user, and appends the user's information to the 
    existing data. The user's approved limit is calculated based on their 
    monthly salary.

    Parameters:
    - first_name (str): First name of the customer.
    - last_name (str): Last name of the customer.
    - age (int): Age of the customer (must be at least 18).
    - phone_no (str): Phone number of the customer.
    - monthly_salary (float): Monthly salary of the customer.

    Returns:
    None: The function saves the updated customer data to the CSV file.
    """
    monthly_salary = int(monthly_salary)
    users = pd.read_csv("dummy data/customer_data.csv")

    if not users.empty:
        next_customer_id = users['CustomerID'].max() + 1
    else:
        next_customer_id = 1  
    
    new_user = pd.DataFrame(
        {
            "CustomerID": [next_customer_id],
            "FirstName": [first_name],
            "LastName": [last_name],
            "Age": [age],
            "PhoneNumber": [phone_no],
            "MonthlySalary": [monthly_salary],
            "ApprovedLimit": [floor(36 * monthly_salary / 100000) * 100000],
        }
    )

    users = pd.concat([users, new_user], ignore_index=True)

    users.to_csv('dummy data/customer_data.csv', index=False)


def get_customer_id(phone_no):
    """
    Retrieve the CustomerID associated with a given phone number.

    This function searches the customer data CSV file for the specified 
    phone number and returns the corresponding CustomerID. If the phone 
    number is not found or the file is not found, the function returns None.

    Parameters:
    - phone_no (str): The phone number of the customer whose ID is to be retrieved.

    Returns:
    - int or None: The CustomerID of the customer if found; otherwise, None.
    """
    try:
        users = pd.read_csv("dummy data/customer_data.csv")
        print(type(users['PhoneNumber']))
    except FileNotFoundError:
        return None  

    customer = users[users['PhoneNumber'] == phone_no]
    
    if not customer.empty:
        return customer['CustomerID'].values[0]  
    else:
        return None


def get_customer_info(phone_no):
    """
    Retrieve the customer information associated with a given phone number.

    This function searches the customer data CSV file for the specified 
    phone number and returns the corresponding customer information as a 
    pandas Series. If the phone number is not found or the file is not 
    found, the function returns None.

    Parameters:
    - phone_no (str): The phone number of the customer whose information is to be retrieved.

    Returns:
    - pandas.Series or None: A Series containing the customer information if found; otherwise, None.
    """
    try:
        users = pd.read_csv("dummy data/customer_data.csv")
    except FileNotFoundError:
        return None
    phone_no = int(phone_no)  

    customer = users[users['PhoneNumber'] == phone_no]

    if not customer.empty:
        return customer.iloc[0]  
    else:
        return None


def calculate_credit_score(customer_data, customer_loan_data):
    """
    Calculate credit score based on multiple factors with weighted importance:
    1. Approved limit usage (25%)
    2. EMI payment history (30%)
    3. Average loan tenure (15%)
    4. Customer history length (10%)
    5. Number of loans (20%)
    """
    today = datetime.now()

    customer_data_df = pd.DataFrame.from_records(customer_data)    
    customer_loan_data_df = pd.DataFrame.from_records(customer_loan_data)
    
    # Ensure date columns are in datetime format
    try:
        # Convert date columns, handling potential format issues
        customer_loan_data_df['DateofApproval'] = pd.to_datetime(customer_loan_data_df['DateofApproval'], format='%Y-%m-%d', errors='coerce')
        customer_loan_data_df['EndDate'] = pd.to_datetime(customer_loan_data_df['EndDate'], format='%Y-%m-%d', errors='coerce')
        
        # Handle any rows where date conversion failed
        if customer_loan_data_df['DateofApproval'].isna().any() or customer_loan_data_df['EndDate'].isna().any():
            print("Warning: Some date conversions failed. Check date formats in data.")
    except Exception as e:
        print(f"Error converting dates: {e}")
        return 0, ["Error processing loan dates"]
    
    customer_active_loans = customer_loan_data_df[customer_loan_data_df['EndDate'] > today]
    
    credit_score = 0
    warning = []
    
    # 1. Check approved limit usage (25 points)
    try:
        perc_amount_vs_limit = (customer_active_loans['LoanAmount'].sum() / customer_data_df['ApprovedLimit'].iloc[0] * 100)
        if perc_amount_vs_limit < 15:
            credit_score += 25
        elif perc_amount_vs_limit < 30:
            credit_score += 22
        elif perc_amount_vs_limit < 35:
            credit_score += 16
        elif perc_amount_vs_limit < 40:
            credit_score += 14
        elif perc_amount_vs_limit < 55:
            credit_score += 10
        elif perc_amount_vs_limit < 60:
            credit_score += 6
        elif perc_amount_vs_limit < 80:
            credit_score += 4
            warning.append("You are very close to your approved limit")
        elif perc_amount_vs_limit < 95:
            credit_score += 2
            warning.append("You are very close to your approved limit")
        else:
            warning.append("You have exceeded your approved limit")
    except Exception as e:
        print(f"Error calculating limit usage: {e}")
        credit_score += 0
    
    # 2. Check EMI payment history (30 points)
    try:
        months_since_approval = ((today.year - customer_loan_data_df['DateofApproval'].dt.year) * 12 + 
                               (today.month - customer_loan_data_df['DateofApproval'].dt.month))
        
        total_months = months_since_approval.sum()
        total_emis_paid = customer_loan_data_df['EMIsPaidOnTime'].sum()
        
        if total_months > 0:
            perc_emi_paid_on_time = (total_emis_paid / total_months) * 100
            
            if perc_emi_paid_on_time >= 120:
                credit_score += 30
            elif perc_emi_paid_on_time >= 100:
                credit_score += 26
            elif perc_emi_paid_on_time >= 90:
                credit_score += 22
            elif perc_emi_paid_on_time >= 80:
                credit_score += 18
            elif perc_emi_paid_on_time >= 70:
                credit_score += 15
            elif perc_emi_paid_on_time >= 60:
                credit_score += 12
            elif perc_emi_paid_on_time >= 50:
                credit_score += 8
                warning.append("Less than 50% EMIs paid on time")
            else:
                credit_score += 4
                warning.append(f"Only {perc_emi_paid_on_time:.1f}% EMIs paid on time")
    except Exception as e:
        print(f"Error calculating EMI history: {e}")
        credit_score += 0
    
    # 3. Average tenure (15 points)
    try:
        avg_tenures = customer_loan_data_df['Tenure'].mean() / 12
        if avg_tenures >= 15:
            credit_score += 15
        elif avg_tenures >= 10:
            credit_score += 12
        elif avg_tenures >= 8:
            credit_score += 8
        elif avg_tenures >= 6:
            credit_score += 6
        elif avg_tenures >= 4:
            credit_score += 4
        elif avg_tenures >= 2:
            credit_score += 2
        else:
            credit_score += 1
    except Exception as e:
        print(f"Error calculating average tenure: {e}")
        credit_score += 0
    
    # 4. Customer history length (10 points)
    try:
        oldest_loan_year = customer_loan_data_df['DateofApproval'].min().year
        if oldest_loan_year <= 2010:
            credit_score += 10
        elif oldest_loan_year < 2014:
            credit_score += 8
        elif oldest_loan_year < 2016:
            credit_score += 5
        elif oldest_loan_year < 2020:
            credit_score += 2
    except Exception as e:
        print(f"Error calculating customer history: {e}")
        credit_score += 0
        
    # 5. Number of loans (20 points)
    try:
        number_of_loans_taken = len(customer_loan_data_df)
        if number_of_loans_taken > 15:
            credit_score += 20
        elif number_of_loans_taken >= 10:
            credit_score += 15
        elif number_of_loans_taken >= 6:
            credit_score += 12
        elif number_of_loans_taken >= 2:
            credit_score += 8
        else:
            credit_score += 2
    except Exception as e:
        print(f"Error calculating number of loans: {e}")
        credit_score += 0
    
    return credit_score, warning


def calculate_monthly_installment(loan_amount, interest_rate, tenure):
    """
    Calculate the monthly installment for a loan.

    This function computes the monthly payment required to repay a loan 
    based on the loan amount, annual interest rate, and loan tenure. 
    It uses the formula for calculating an annuity payment.

    Parameters:
    - loan_amount (float): The total amount of the loan.
    - interest_rate (float): The annual interest rate (as a percentage).
    - tenure (int): The loan tenure in months.

    Returns:
    - float: The calculated monthly installment.
    """
    monthly_interest_rate = interest_rate / 12 / 100
    total_months = tenure
    monthly_installment = (loan_amount * monthly_interest_rate) / (1 - (1 + monthly_interest_rate) ** -total_months)

    return monthly_installment



def emis_exceed_limit(customer_data, customer_loan_data, current_loan_installment):
    """
    Determine if the total EMIs (Equated Monthly Installments) exceed 
    50% of the customer's monthly salary.

    This function checks if the sum of the current EMIs for active loans 
    along with a new loan installment exceeds half of the customer's 
    monthly salary. It is used to assess the affordability of new loans 
    for the customer.

    Parameters:
    - customer_data (list of dict): A list containing customer information,
      including 'MonthlySalary'.
    - customer_loan_data (list of dict): A list containing loan information,
      including 'EndDate' and 'Monthlypayment'.
    - current_loan_installment (float): The EMI of the current loan being 
      considered for approval.

    Returns:
    - bool: True if the total EMIs exceed 50% of the monthly salary, 
      False otherwise.
    """
    today = datetime.now()

    customer_loan_data_df = pd.DataFrame.from_records(customer_loan_data)
    customer_data_df = pd.DataFrame.from_records(customer_data)

    customer_loan_data_df['EndDate'] = pd.to_datetime(customer_loan_data_df['EndDate'])
    customer_active_loans = customer_loan_data_df[customer_loan_data_df['EndDate'] > today]

    monthly_salary = customer_data_df.iloc[0]['MonthlySalary']
    sum_of_current_emis = customer_active_loans['Monthlypayment'].sum()

    return (sum_of_current_emis + current_loan_installment) > (monthly_salary * 0.5)


def get_eligibility(credit_score, interest_rate):
    """
    Determine the loan eligibility of a customer based on their credit score
    and the applicable interest rate.

    This function assesses whether a customer qualifies for a loan and, if 
    not, provides reasons for rejection. It also corrects the interest rate 
    based on the customer's credit score.

    Parameters:
    - credit_score (int): The credit score of the customer, on a scale from 0 to 100.
    - interest_rate (float): The proposed interest rate for the loan, in percentage.

    Returns:
    - bool: True if the customer is eligible for the loan, False otherwise.
    - float: The corrected interest rate based on eligibility criteria.
    - list: A list of reasons for rejection if the customer is not eligible.
    """
    rejected_reason = []
    
    approval = True
    
    if credit_score > 50:
        corrected_interest_rate = interest_rate
        approval = True
    elif 30 < credit_score <= 50:
        if interest_rate < 12:
            approval = False
            rejected_reason.append("You are only eligible for interest rates above 12%")
        corrected_interest_rate = max(interest_rate, 12)
    elif 10 < credit_score <= 30:
        if interest_rate < 16:
            approval = False
            rejected_reason.append("You are only eligible for interest rates above 16%")
        corrected_interest_rate = max(interest_rate, 16)
    else:
        approval = False
        rejected_reason.append("Your credit score is too low.")
        corrected_interest_rate = None
        
    return approval, corrected_interest_rate, rejected_reason

import random

def generate_unique_loan_id(loan_data):
    """
    Generate a unique loan ID that does not exist in the given loan data.

    This function continuously generates random loan IDs in the range 
    from 100 to 10000 until it finds one that is not already present 
    in the provided loan data.

    Parameters:
    - loan_data (pd.DataFrame): A DataFrame containing existing loan records, 
      specifically with a column 'LoanID' that holds the IDs of current loans.

    Returns:
    - int: A unique loan ID that is not already present in the loan data.
    """
    while True:
        loan_id = random.randint(100, 10000)  
        if loan_id not in loan_data['LoanID'].values:
            return loan_id

def create_loan(phone_no, loan_amount, interest_rate, tenure):
    """
    Create a loan for a customer based on their phone number and loan parameters.

    This function checks if the customer exists, verifies their credit score,
    eligibility for the loan, and ensures that the monthly installment does not
    exceed 50% of their monthly salary. If all conditions are met, a new loan 
    entry is created and saved to the loan data file.

    Parameters:
    - phone_no (str): The customer's phone number to identify the customer.
    - loan_amount (float): The amount of the loan requested by the customer.
    - interest_rate (float): The interest rate of the loan.
    - tenure (int): The duration of the loan in months.

    Returns:
    - dict: A dictionary containing:
        - 'loan_data': A nested dictionary with details about the loan including:
            - loan_id (int or None): The unique ID of the loan if approved.
            - customer_id (int): The ID of the customer.
            - loan_approved (bool): Indicates if the loan was approved.
            - message (list): Reasons for rejection if not approved.
            - monthly_installment (float or None): The calculated monthly installment if approved.
        - 'message' (str): A success or failure message regarding loan creation.
    
    Raises:
    - FileNotFoundError: If the customer or loan data CSV files do not exist.
    """

    today = datetime.now()

    # Load the customer data
    customers = pd.read_csv("dummy data/customer_data.csv")

    # Fetch the customer ID using the phone number
    phone_no = int(phone_no)
    customer = customers[customers['PhoneNumber'] == phone_no]

    if customer.empty:
        return {'message': "No such user exists"}, 404

    customer_id = customer['CustomerID'].values[0]

    # Check existing loans
    loans = pd.read_csv("dummy data/loan_data.csv")

    # Generate a unique loan ID
    existing_loan_ids = loans['LoanID'].unique()
    loan_id = random.randint(1000, 10000)
    while loan_id in existing_loan_ids:
        loan_id = random.randint(1000, 10000)

    # Calculate credit score and eligibility
    loan_data = loans[loans['CustomerID'] == customer_id]

    approval = True
    rejected_reason = []
    corrected_interest_rate = None

    if not loan_data.empty:
        credit_score, _ = calculate_credit_score(customer.to_dict(orient='records'), loan_data.to_dict(orient='records'))
        approval, corrected_interest_rate, rejected_reason = get_eligibility(credit_score, interest_rate)

        monthly_installment = calculate_monthly_installment(loan_amount, corrected_interest_rate, tenure)

        if emis_exceed_limit(customer.to_dict(orient='records'), loan_data.to_dict(orient='records'), monthly_installment):
            rejected_reason.append("Sum of all your EMIs exceeds 50% of your monthly salary.")
            approval = False
    else:
        if loan_amount > 1_000_000:
            approval = False
            rejected_reason.append("Since you have no credit history, you are not eligible for loans exceeding amount 1,000,000.")

        if interest_rate < 12:
            approval = False
            rejected_reason.append("Since you have no credit history, you are not eligible for loans with Interest Rates less than 12%.")

    if approval:
        monthly_installment = calculate_monthly_installment(loan_amount, interest_rate, tenure)
        # Save approved loan to CSV
        new_loan = pd.DataFrame({
            "LoanID": [loan_id],
            "CustomerID": [customer_id],
            "LoanAmount": [loan_amount],
            "Tenure": [tenure],
            "InterestRate": [corrected_interest_rate or interest_rate],
            "MonthlyPayment": [monthly_installment],
            "EMIsPaidOnTime": [0],
            "DateOfApproval": [today.strftime('%Y-%m-%d')],
            "EndDate": [(today + relativedelta(months=tenure)).strftime('%Y-%m-%d')]
        })

        loans = pd.concat([loans, new_loan], ignore_index=True)
        loans.to_csv('dummy data/loan_data.csv', index=False)

        loan_id = loan_id  # Return the unique loan ID
    else:
        loan_id = None

    return {
        'loan_data': {
            "loan_id": loan_id,
            "customer_id": customer_id,
            "loan_approved": approval,
            "message": rejected_reason,
            "monthly_installment": monthly_installment if approval else None
        },
        'message': "Loan created successfully" if approval else "Loan creation failed"
    }, 200


def view_loans(phone_number, columns=['LoanAmount', 'Tenure', 'InterestRate', 'EndDate', 'MonthlyPayment', 'DateOfApproval']):
    """
    Retrieve and display loan information for a customer based on their phone number.

    This function checks if a customer exists by their phone number, retrieves their 
    loans if available, and filters the loan details based on the specified columns.

    Parameters:
    - phone_number (str): The phone number of the customer to look up.
    - columns (list of str): A list of column names to include in the response. 
                             Defaults to a standard set of loan attributes.

    Returns:
    - dict: A dictionary containing:
        - 'customer_id' (int): The ID of the customer.
        - 'loans' (list): A list of dictionaries with loan details if found; 
                          otherwise, returns a message indicating no loans.
    - int: HTTP status code indicating the result of the request (200 for success, 404 for not found).

    Raises:
    - FileNotFoundError: If the customer or loan data CSV files do not exist.
    """

    customer_data = pd.read_csv('dummy data/customer_data.csv')
    phone_number = int(phone_number)

    customer_row = customer_data[customer_data['PhoneNumber'] == phone_number]

    if customer_row.empty:
        return {"message": "No such customer exists"}, 404  

    customer_id = customer_row.iloc[0]['CustomerID']  

    loan_data = pd.read_csv('dummy data/loan_data.csv')

    loans = loan_data[loan_data['CustomerID'] == customer_id]
    if columns:
        loans = loans[columns]

    if loans.empty:
        return {"message": "No loans found for this customer"}, 404  

    loans_list = loans.to_dict(orient='records')

    return {
        "customer_id": customer_id,
        "loans": loans_list
    }, 200



def check_eligibility(phone_no, loan_amount, interest_rate, tenure):
    """
    Check the eligibility of a customer for a loan based on their phone number, 
    loan amount, interest rate, and tenure.

    This function retrieves customer information and existing loan data, 
    calculates the credit score, determines eligibility, and computes the 
    monthly installment for the requested loan.

    Parameters:
    - phone_no (str): The phone number of the customer.
    - loan_amount (float): The amount of the loan being requested.
    - interest_rate (float): The proposed interest rate for the loan.
    - tenure (int): The tenure for which the loan is requested (in months).

    Returns:
    - dict: A dictionary containing:
        - 'data' (dict): A dictionary with eligibility details, including:
            - 'credit_score' (float): The calculated credit score of the customer.
            - 'customer_id' (int): The ID of the customer.
            - 'approval' (bool): Indicates if the customer is approved for the loan.
            - 'interest_rate' (float): The proposed interest rate.
            - 'corrected_interest_rate' (float): The final interest rate based on eligibility.
            - 'tenure' (int): The tenure for the loan.
            - 'monthly_installment' (float): The calculated monthly installment.
            - 'reason_for_rejection' (list): Reasons for rejection, if applicable.
            - 'warnings' (list): Any warnings regarding eligibility.
        - 'message' (str): A message indicating eligibility status.

    - int: HTTP status code (200 for eligible, 403 for not eligible).

    Raises:
    - FileNotFoundError: If the customer or loan data CSV files do not exist.
    """

    customer_data = pd.read_csv('dummy data/customer_data.csv')
    loan_data = pd.read_csv('dummy data/loan_data.csv')
    phone_no = int(phone_no)
    
    # Find the customer by phone number
    customer_row = customer_data[customer_data['PhoneNumber'] == phone_no]

    if customer_row.empty:
        return {"message": "No such user exists"}, 404  

    customer_id = customer_row.iloc[0]['CustomerID']  

    # Check if the user has loan data
    user_loans = loan_data[loan_data['CustomerID'] == customer_id]
    # if user_loans.empty:
    #     return {"message": "User has no credit history"}, 404 

    # Convert to records to pass into the function
    credit_score, warning = calculate_credit_score(customer_row.to_dict(orient='records'), user_loans.to_dict(orient='records'))

    approval, corrected_interest_rate, rejected_reason = get_eligibility(credit_score, interest_rate)

    monthly_installment = calculate_monthly_installment(loan_amount, corrected_interest_rate, tenure)

    if emis_exceed_limit(customer_row.to_dict(orient='records'), user_loans.to_dict(orient='records'), monthly_installment):
        rejected_reason.append("Sum of all your EMIs exceeds 50% of your monthly salary.")
        approval = False

    return {
        'data': {
            "credit_score": credit_score,
            "customer_id": customer_id,
            "approval": approval,
            "interest_rate": interest_rate,
            "corrected_interest_rate": corrected_interest_rate,
            "tenure": tenure,
            "monthly_installment": monthly_installment,
            "reason_for_rejection": rejected_reason or None,
            "warnings": warning
        },
        'message': "Eligible for loan" if approval else "Not Eligible for loan"
    }, 200 if approval else 403  # HTTP status codes

st.set_page_config(page_title="Loan Assistant", page_icon="💰")

openai.api_key = st.secrets["OPEN_AI"]
llm = OpenAI(model="gpt-4o", temperature=0.2)

tools = [
    # FunctionTool.from_defaults(fn=add_user),
    FunctionTool.from_defaults(fn=get_customer_info),
    FunctionTool.from_defaults(fn=create_loan),
    FunctionTool.from_defaults(fn=view_loans),
    FunctionTool.from_defaults(fn=check_eligibility)
]

st.sidebar.header("Create User")

with st.sidebar.form(key='create_user_form'):
    firstName = st.text_input("First Name")
    lastName = st.text_input("Last Name")
    Age = st.text_input("Age")
    phone_number = st.text_input("phone no")
    montly_Salary = st.text_input("monthly salary")
    submit_button = st.form_submit_button("Create User")

    if submit_button:
        add_user(first_name=firstName,last_name=lastName,age=Age,phone_no=phone_number,monthly_salary=montly_Salary)
        st.sidebar.success("User created successfully!")

agent = ReActAgent.from_tools(tools=tools,llm=llm,verbose=True)
if query := st.chat_input("How can I help you with your loan today?"):
        st.chat_message("user").markdown(query)
        
        # Create context-aware prompt
        prompt = f"""You are a helpful loan assistant.
        Please help with their query: {query}
        
        Available actions:
        1. Check loan eligibility using check_loan_eligibility()
        2. View existing loans using get_user_loans()
        3. Create new loans using create_new_loan()
        4. Get user details using get_user_details()
        5. You can also add user add_user()
        
        If the query involves loan amounts or terms, you can ask for specific details if needed.
        
        Important guidelines:
        - Always verify user details before providing sensitive information
        - For new loan requests, ask for amount, tenure, and preferred interest rate if not provided
        - Explain eligibility criteria and reasons for rejection clearly
        - Provide monthly installment calculations when relevant
        """
        
        response = agent.chat(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)


# Footer
st.markdown("---")
st.markdown("*This is an AI-powered loan assistant. For urgent matters, please contact our customer service.*")
