# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:18:44 2021

@author: Sagun Shakya
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html', title = 'Sample')

@app.route("/about", methods = ["GET", "POST"])
def some_function():
    if request.method == "POST":
        first_name = request.form.get("FIRSTNAME")    # Use name attribute in HTML.
        last_name = request.form.get("LASTNAME")
        age = request.form.get("AGE")
        gender = request.form.get("GENDER")
        return f"<h2> Your full name is {first_name} {last_name} and your age is {age} years. Your are a {gender}.</h2>"
    return render_template('about.html')
    

@app.route("/predict", methods = ['GET', 'POST'])
def predict_income():
    if request.method == 'POST':
        get_data = lambda attribute: request.form.get(attribute)
        
        # CATEGORICAL VARIABLES.
        workclass = 'workclass_' + get_data("WORKCLASS")
        education = 'education_' + get_data("EDUCATION")
        marital_status = 'marital-status_' + get_data("MARITAL-STATUS")
        occupation = 'occupation_' + get_data("OCCUPATION")
        country = 'country_' + get_data("COUNTRY")           # Can ignore this.
        race = 'race_' + get_data("RACE")
        gender = 'gender_' + get_data("GENDER")
        
        # Numerical variables.
        age = get_data("AGE")
        hours_per_week = get_data("HOURS-PER-WEEK")
        capital_gain = get_data("CAPITAL-GAIN")
        capital_loss = get_data("CAPITAL-LOSS")
        
        # Processing the numerical variables.
        scaler = joblib.load('minmax_scaler.gz')        
        numerical_data = {
            'age': [int(age)],
            'capital-gain': [int(capital_gain)],
            'capital-loss': [int(capital_loss)],
            'hours-per-week': [int(hours_per_week)]
            }

        df_numerical = pd.DataFrame(numerical_data, columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week'])
        numerical_data = scaler.transform(df_numerical)
        
        # Processing the categorical variables.
        to_ignore = ('workclass_Never-worked', 'education_Assoc-acdm', 
                     'marital-status_Married', 'occupation_Adm-clerical',
                     'race_Amer-Indian-Eskimo', 'gender_Female')
        
        options = ['workclass_Private',
         'workclass_Without-pay',
         'workclass_govt_employees',
         'workclass_self_employed',
         'education_Assoc-voc',
         'education_Bachelors',
         'education_Doctorate',
         'education_HS-grad',
         'education_Masters',
         'education_Preschool',
         'education_Prof-school',
         'education_Some-college',
         'education_elementary_school',
         'marital-status_Never-married',
         'marital-status_Separated',
         'marital-status_Widowed',
         'occupation_Armed-Forces',
         'occupation_Craft-repair',
         'occupation_Exec-managerial',
         'occupation_Farming-fishing',
         'occupation_Handlers-cleaners',
         'occupation_Machine-op-inspct',
         'occupation_Other-service',
         'occupation_Priv-house-serv',
         'occupation_Prof-specialty',
         'occupation_Protective-serv',
         'occupation_Sales',
         'occupation_Tech-support',
         'occupation_Transport-moving',
         'race_Asian-Pac-Islander',
         'race_Black',
         'race_Other',
         'race_White',
         'gender_Male']
        
        data_dict = {column : [0] for column in options}
        df_categorical = pd.DataFrame(data_dict)
        
        # Updating with ones.
        categorical_variables = [workclass, education, marital_status, occupation, race, gender]
        
        for var in categorical_variables:
            if var not in to_ignore:
                df_categorical[var] = 1
            
        # Merging the numerical and categorical variables to get the final input.
        df_input = pd.concat([df_numerical, df_categorical], axis = 1)
        
        # Feeding the input to the classification model.
        model = joblib.load('logistic_regression_model.pkl')
        Y = model.predict(df_input)[0]
        
        # Interpretation.
        if Y == 1:
            prediction = 'greater than 50k'
        else:
            prediction = 'less than or equal to 50k'

        #return f"Based on the input data, your income is {prediction}."
        
        return render_template('predict.html', prediction = prediction)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug = True)

