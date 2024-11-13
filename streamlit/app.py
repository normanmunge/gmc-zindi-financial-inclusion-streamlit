import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

model  =  joblib.load(open('joblib-files/rm-model.joblib', 'rb'))
encoder_dict = joblib.load(open('joblib-files/encoder.joblib', 'rb'))

def load_data():
    return pd.read_csv('datasets/financial_inclusion_dataset.csv')

def main():
    html_title_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Account Status Prediction App </h2>
    </div>
    """
    st.markdown(html_title_temp, unsafe_allow_html = True)

    html_body_temp = """
    <div style = "margin: 0 1.875rem></div>
    """

    st.markdown(html_body_temp, unsafe_allow_html = True)

    df = load_data()

    age_of_respondent = st.text_input("Age","0")
    country = st.selectbox("Country", df['country'].unique().tolist())
    education_level = st.selectbox("Education", df['education_level'].unique().tolist()) 
    gender_of_respondent = st.selectbox("Gender",df['gender_of_respondent'].unique().tolist())
    job_type = st.selectbox("Job Type",df['job_type'].unique().tolist())
    cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])

    if st.button("Predict"):
        data = {
            'age_of_respondent': int(age_of_respondent),
            'country': country,
            'education_level': education_level,
            'gender_of_respondendt': gender_of_respondent,
            'job_type': job_type,
            'cellphone_access': cellphone_access
        }

        st_df = pd.DataFrame([list(data.values())], columns=['age_of_respondent','country','education_level','gender_of_respondent','job_type','cellphone_access'])
       
        st_df['gender_of_respondent'] = st_df['gender_of_respondent'].replace({'Male': 1, 'Female': 2})
        st_df['cellphone_access'] = st_df['cellphone_access'].replace({'Yes': 1, 'No': 2})

        for cat in encoder_dict:
            for col in st_df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = encoder_dict[cat]
                    st_df[col] = le.fit_transform(st_df[col])

        features_list = st_df.values.tolist()      
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 1:
            text = "✅"
        else:
            text = "❌"

        st.success('Has bank account {}'.format(text))


if __name__=='__main__': 
    main()
