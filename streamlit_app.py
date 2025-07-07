import pandas as pd
import numpy as np
import json
import boto3
import streamlit as st
import requests

global app_name
global region
app_name = 'my-deployment-attemp'
region = 'us-east-1'

global answers_tab
answers_tab = ["sans risque", "à risque"]

def check_status(app_name):
    sage_client = boto3.client('sagemaker', region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description['EndpointStatus']
    return endpoint_status

def query_endpoint(app_name, input_json):
	client = boto3.session.Session().client('sagemaker-runtime', region)

	response = client.invoke_endpoint(
		EndpointName = app_name,
		Body = input_json,
		ContentType = 'application/json'#'; format=pandas-split',
		)

	preds = response['Body'].read().decode('ascii')
	preds = json.loads(preds)
	#print('Received response: {}'.format(preds))
	return preds


def main():
	MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
	
	st.title('Prédiction du risque de crédit')
	
	status_btn = st.button('Status')
	if status_btn:
		st.write(
            'Application status is {}'.format(check_status(app_name)))

	birth_years = st.number_input('Age - [20, 70]',
                                 min_value=20., max_value=70., value=45., step=1.)

	days_employed_input = st.number_input('Années en emploi, [0, 50]',
                                 min_value=0., max_value=50., value=5., step=1.)
	days_employed = days_employed_input*(-365.)

	days_last_phone_change_input = st.number_input('Dernier changement de téléphone (années) - [0, 10]',
                                 min_value=0., max_value=10., value=5., step=2.)
	days_last_phone_change = days_last_phone_change_input*(-365.)

	amt_credit_input = st.number_input('Crédit total (en centaine de millier) - [0.5, 40]',
                                 min_value=0.5, max_value=40., value=5., step=1.)
	amt_credit = amt_credit_input*10000.

	amt_annuity_input = st.number_input('Montant annuel (en millier) - [2, 250]',
                                 min_value=2., max_value=250., value=25., step=1.)
	amt_annuity = amt_annuity_input*1000.

	ext_source_2 = st.number_input('Source extérieur (2) - [0, 0.9]',
                                 min_value=0., max_value=0.9, value=0.5, step=0.1)

	ext_source_3 = st.number_input('Source extérieur (3) - [0, 0.9]',
                                 min_value=0., max_value=0.9, value=0.5, step=0.1)

	predict_btn = st.button('Prédire')
	if predict_btn:
		arr_predict = [ext_source_2,
						ext_source_3,
						days_employed,
						birth_years,
						amt_credit,
						amt_annuity,
						days_last_phone_change]
		query_input = pd.DataFrame(arr_predict).transpose().to_dict(orient='split')
		data = {"dataframe_split": query_input}
		byte_data = json.dumps(data).encode('utf-8')
		
		pred = query_endpoint(app_name=app_name, input_json=byte_data)
		
		st.write('Le modèle prédit que le client est {}'.format(answers_tab[pred['predictions'][0]]))


if __name__ == '__main__':
    main()
