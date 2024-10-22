import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, classification_report, ConfusionMatrixDisplay

import time

# load the data
def load_data():
    return pd.read_csv('../datasets/financial_inclusion_dataset.csv')


# Handling outliers using the IQR method
# def outlier_limits(col):
#     Q3, Q1 = np.nanpercentile(col, [75,25])
#     inter_quartile_range = Q3 - Q1
#     upper_limit = Q3 + (1.5*inter_quartile_range)
#     lower_limit = Q1 - (1.5*inter_quartile_range)
#     return upper_limit, lower_limit

# def handle_outliers(df):
#     outlier_cols = df.drop(df.select_dtypes(include=['object']).columns.tolist(), axis=1)
#     outlier_cols.drop('churn', axis=1, inplace=True) #Let's drop churn since it will be our target column
#     outlier_cols.columns.tolist()

#     for col in outlier_cols:
#         UL, LL = outlier_limits(df[col])
#         df.loc[(df[col] > UL), col] = UL
#         df.loc[(df[col] < LL), col] = LL

#     return df

# Encoding our dataset
def encode_data(df):
    encoder = LabelEncoder()

    # Let's replace the gender with 1 - Male & 2 - Female
    df['gender_of_respondent'] = df['gender_of_respondent'].replace({'Male': 1, 'Female': 2})

    # Let's replace the bank account with 1 - Yes & 2 - No
    df['bank_account'] = df['bank_account'].replace({'Yes': 1, 'No': 2})

    # Let's replace the cellphone access with 1 - Yes & 2 - No
    df['cellphone_access'] = df['cellphone_access'].replace({'Yes': 1, 'No': 2})

    categorical_cols = ['country','location_type','year', 
                    'relationship_with_head', 'marital_status', 
                    'education_level', 'job_type'
                   ]

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df

# Model Training
def train_model(df):
    X = df[['education_level', 'gender_of_respondent', 'job_type', 'age_of_respondent', 'cellphone_access', 'country']]
    y = df['bank_account']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training the model
    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)

    # Making predictions
    y_pred = cls.predict(X_test)

    # TODO:-> Adding the user_id column to the prediction_df
    #X_test['user_id'] = df.user_id

    # Return the model and y_pred variables
    return cls, y_pred, y_test
    #prediction_proba = model.predict_proba(X_test)


def evaluate_model(model, y_pred, y_test):
    # prediction_proba = model.predict_proba(test_data)
    mae = mean_absolute_error(y_test, y_pred)

    st.header('Bank account Prediction')

    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    # TODO:-> Adding the user_id column to the prediction_df
    #prediction_df['user_id'] = test_data.index
    return pred_df


def preprocess_data(df):

    with st.expander('Data'):
        st.write(df.head())

    # Handling outliers in our dataset
    #df = handle_outliers(df)

    # Encoding the dataset
    df = encode_data(df)

    model, y_pred, y_test = train_model(df)

    return evaluate_model(model, y_pred, y_test)


def define_input_handlers(df, features):
    options = []

    for feature in features:
        unique_values = df[feature].unique().tolist()
        feature_name = f'{feature.capitalize().replace("_", " ")}'
        unique_values = df[feature].unique().tolist()
        option = st.selectbox(feature_name, unique_values)
        options.append(option)

    if 'processed' not in st.session_state:
        st.session_state.processed = {}

    if st.button('Predict'):
        for option in options:
            result = show_computation_progress(option)
            st.session_state.processed[option] = result

        if option in st.session_state.processed:
            st.write(st.session_state.processed[option][0])


def show_computation_progress(option):

    with st.spinner('Loading...'):
        time.sleep(.5)

    return f'{option} processed'

    # 'Loading...'

    # # Add a placeholder
    # latest_iteration = st.empty()
    # bar = st.progress(0)

    # for i in range(100):
    # # Update the progress bar with each iteration.
    #     latest_iteration.text(f'{i+1} Percentage')
    #     bar.progress(i + 1)
    #     time.sleep(0.1)

def main():
    st.set_page_config(
        page_title = 'Bank Account Status Prediction',
        layout = 'centered'
    )

    st.title('Predict whether people own or use bank account')

    # Load the data -> load_data()
    df = load_data()

    # Preprocess the data, train the model and evaluate the model
    predict = preprocess_data(df)

    st.dataframe(predict, width=1000, height=500)

    # Define input parameters in the sidebar
    with st.sidebar:
        st.header('Input features')

        st.write('Update parameters to see the prediction')
        
        # Define the input features
        features = ['education_level', 'gender_of_respondent', 'job_type', 'age_of_respondent', 'cellphone_access', 'country']

        unique_values = define_input_handlers(df, features)

    


main()


