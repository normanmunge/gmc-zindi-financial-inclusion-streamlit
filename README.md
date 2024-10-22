# Predicting Financial Inclusion in Africa

Financial inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 14% of adults) have access to or use a commercial bank account.

Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and make payments while also helping businesses build up their credit-worthiness and improve their access to loans, insurance, and related services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.

The objective of the (Zindi challenge)[https://zindi.africa/competitions/financial-inclusion-in-africa] is to create a machine learning model to predict which individuals are most likely to have or use a bank account. The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, while providing insights into some of the key factors driving individuals’ financial security.

To access the dataset use this [link](https://drive.google.com/file/d/1FrFTfUln67599LTm2uMTSqM8DjqpAaKL/view), otherwise it's included in the datasets directory on this repository.

### About the repository

This repository contains 1 [jupyter notebook](https://github.com/normanmunge/gmc-zindi-financial-inclusion-streamlit/blob/main/financial-inclusion-prediction.ipynb) and a [streamlit app](https://github.com/normanmunge/gmc-zindi-financial-inclusion-streamlit/tree/main/streamlit) to showcase the predictions on a web interface.

It is a basic app using Streamlit to showcase for my portfolio. Interested parties can fork it and improve on it.

### Running the notebook

The instructions below show how to run the notebooks on a mac terminal.

To run the notebook. Follow the instructions below on a terminal:

#### A. Create a virtual environment

A virtual environment is recommended to install the necessary libraries needed to run the notebook. A virtual environment separates your local projects libraries and the global machine's libraries reducing the chances of conflicts.

`python -m venv [path/to/environments/name-of-env]`

#### B. Activate your virtual environment

`source [path/to/environment]/bin/activate`

#### C. Install necessary libraries

`pip install numpy pandas matplotlib seaborn statsmodels jupyterlab`

#### D. Running the notebook

`jupyter lab`

#### E. Saving and closing your virtual environment

To save your projects, `Cmd + S` will do the trick. Afterwards, on your terminal, enter `deactivate`
