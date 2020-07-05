import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from opioid_functions import *

# read the data
data = pd.read_csv("npi_outliers.csv")

# subset to US
data = data[data['nppes_provider_country'] == 'US']

# look at residuals by state
df_state = data[['nppes_provider_state', 'diff']].groupby('nppes_provider_state').mean().reset_index()
df_state = df_state[~df_state.nppes_provider_state.isin(['MP', 'XX', 'AA', 'GU'])]

# plot state results
plt.rcParams['figure.figsize'] = (14, 10)
sns.barplot(
    x='diff',
    y='nppes_provider_state',
    data=df_state.sort_values('diff', ascending=False)
).set_title('Average Prescription Rate Residuals by State')
plt.show()

# look at residuals by credentials
df_cred = data[
    [
        'credential_md',
        'credential_rn', 
        'credential_phd',
        'credential_dds',
        'credential_pa',
        'credential_mba',
        'diff'
    ]
]

df_cred['credentials'] = df_cred.iloc[:, :-1].idxmax(axis=1)
df_cred = df_cred[['credentials', 'diff']].groupby('credentials').mean().reset_index()

# plot credential results
plt.rcParams['figure.figsize'] = (14, 10)
sns.barplot(
    x='diff',
    y='credentials',
    data=df_cred.sort_values('diff', ascending=False)
).set_title('Average Prescription Rate Residuals by Credentials')
plt.show()

# look at residuals by specialty
df_spec = data[
    [
        'specialty_description_cardiology',
        'specialty_description_dentist',
        'specialty_description_dermatology',
        'specialty_description_emergency_medicine',
        'specialty_description_family_practice',
        'specialty_description_gastroenterology', 
        'specialty_description_general_surgery',
        'specialty_description_internal_medicine',
        'specialty_description_neurology',
        'specialty_description_nurse_practitioner',
        'specialty_description_obstetrics__gynecology',
        'specialty_description_ophthalmology',
        'specialty_description_optometry', 
        'specialty_description_orthopedic_surgery',
        'specialty_description_physician_assistant',
        'specialty_description_podiatry',
        'specialty_description_psychiatry',
        'specialty_description_psychiatry__neurology',
        'diff'
    ]
]

df_spec['specialty'] = df_spec.iloc[:, :-1].idxmax(axis=1)
df_spec = df_spec[['specialty', 'diff']].groupby('specialty').mean().reset_index()

# plot credential results
plt.rcParams['figure.figsize'] = (12, 10)
sns.barplot(
    x='diff',
    y='specialty',
    data=df_spec.sort_values('diff', ascending=False)
).set_title('Average Prescription Rate Residuals by Specialty')
plt.show()

[c for c in data.columns]