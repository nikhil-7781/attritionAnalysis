from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)


model = XGBClassifier()
model.load_model('xgboost_model.json')  


with open('./templates/index.html', 'r') as file:
    HTML_CONTENT = file.read()

@app.route('/')
def index():
   
    return render_template_string(HTML_CONTENT)

@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = {}
    for key, value in request.form.items():
        try:
            form_data[key] = [float(value)]
        except ValueError:
            form_data[key] = [value]
    
    employee_df = pd.DataFrame(form_data)

  
    employee_df['BusinessTravel_Non-Travel'] = 1 if employee_df['BusinessTravel'][0] == 'Non-Travel' else 0
    employee_df['BusinessTravel_Travel_Frequently'] = 1 if employee_df['BusinessTravel'][0] == 'Travel_Frequently' else 0
    employee_df['BusinessTravel_Travel_Rarely'] = 1 if employee_df['BusinessTravel'][0] == 'Travel_Rarely' else 0
    employee_df.drop(columns=['BusinessTravel'], inplace=True)

    employee_df['Department_Human Resources'] = 1 if employee_df['Department'][0] == 'Human Resources' else 0
    employee_df['Department_Research & Development'] = 1 if employee_df['Department'][0] == 'Research & Development' else 0
    employee_df['Department_Sales'] = 1 if employee_df['Department'][0] == 'Sales' else 0
    employee_df.drop(columns=['Department'], inplace=True)

    
    employee_df['EducationField_Human Resources'] = 1 if employee_df['EducationField'][0] == 'Human Resources' else 0
    employee_df['EducationField_Life Sciences'] = 1 if employee_df['EducationField'][0] == 'Life Sciences' else 0
    employee_df['EducationField_Marketing'] = 1 if employee_df['EducationField'][0] == 'Marketing' else 0
    employee_df['EducationField_Medical'] = 1 if employee_df['EducationField'][0] == 'Medical' else 0
    employee_df['EducationField_Other'] = 1 if employee_df['EducationField'][0] == 'Other' else 0
    employee_df['EducationField_Technical Degree'] = 1 if employee_df['EducationField'][0] == 'Technical Degree' else 0
    employee_df.drop(columns=['EducationField'], inplace=True)

  
    job_roles = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
                 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive',
                 'Sales Representative']
    for role in job_roles:
        employee_df[f'JobRole_{role}'] = 1 if employee_df['JobRole'][0] == role else 0
    employee_df.drop(columns=['JobRole'], inplace=True)

    employee_df['MaritalStatus_Single'] = 1 if employee_df['MaritalStatus'][0] == 'Single' else 0
    employee_df['MaritalStatus_Married'] = 1 if employee_df['MaritalStatus'][0] == 'Married' else 0
    employee_df['MaritalStatus_Divorced'] = 1 if employee_df['MaritalStatus'][0] == 'Divorced' else 0
    employee_df.drop(columns=['MaritalStatus'], inplace=True)

    employee_df['OverTime_No'] = 1 if employee_df['OverTime'][0] == '0' else 0
    employee_df['OverTime_Yes'] = 1 if employee_df['OverTime'][0] == '1' else 0
    employee_df.drop(columns=['OverTime'], inplace=True)

   
    employee_df['Gender_Female'] = 1 if employee_df['Gender'][0] == 'Female' else 0
    employee_df['Gender_Male'] = 1 if employee_df['Gender'][0] == 'Male' else 0
    employee_df.drop(columns=['Gender'], inplace=True)

   
    employee_df['Over18_Y'] = 1

    columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber',
               'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
               'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
               'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
               'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
               'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Non-Travel',
               'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
               'Department_Research & Development', 'Department_Sales', 'EducationField_Human Resources',
               'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
               'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Female', 'Gender_Male',
               'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
               'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
               'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
               'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single', 'Over18_Y', 'OverTime_No',
               'OverTime_Yes']
    
    employee_df = employee_df[columns]

   
    prediction = model.predict(employee_df)
    probability = model.predict_proba(employee_df)[0, 1] 

    
    probability = float(probability)

    result = {
        "AttritionPrediction": "Yes" if prediction[0] == 1 else "No",
        "AttritionProbability": probability
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
