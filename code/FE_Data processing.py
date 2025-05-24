import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#Load data
filename = 'dataset_MLproject.xlsx'
df= pd.read_excel('dataset_MLproject.xlsx')
df = df.drop(columns=['auditor_name'])
df = df.drop(columns=['AUDIT_OPINION_TYPE'])
df = df.drop(columns=['opinion_fiscal_year_end'])     


#TARGET VALUE BEFORE CLEANING
counts = df['HAS_GOING_CONCERN_MODIFICATION'].value_counts()
print(counts)
plt.bar(counts.index, counts.values, color=['blue', 'orange'], edgecolor='black')
plt.title('Distribution of Y')
plt.xlabel('HAS_GOING_CONCERN_MODIFICATION')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['0', '1'])
plt.show()


# MISSING VALUES#
print(df.isnull().sum())

df = df.dropna(subset=['SIC_CODE_FKEY'])

counts = df['IS_EU_REGULATED_SUBMARKET'].value_counts()
print(counts)
df['IS_EU_REGULATED_SUBMARKET'] = df['IS_EU_REGULATED_SUBMARKET'].fillna(0)

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_TAX_FEES_USD'] = imputer.fit_transform(df[['MOST_RECENT_TAX_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['TOTAL_EMPLOYEES'] = imputer.fit_transform(df[['TOTAL_EMPLOYEES']])

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_OTHER_FEES_USD'] = imputer.fit_transform(df[['MOST_RECENT_OTHER_FEES_USD']])

df['auditor_home_office_state'] = df['auditor_home_office_state'].fillna('Unknown')

df['opinion_signature_date'] = pd.to_datetime(df['opinion_signature_date'], errors='coerce').dt.year
df['opinion_signature_date'] = df['opinion_signature_date'].fillna(-1)
df['IPO_YEAR'] = pd.to_datetime(df['IPO_DATE'], errors='coerce').dt.year
df['IPO_YEAR'] = df['IPO_YEAR'].fillna(-1)
df = df.drop(columns=['IPO_DATE','opinion_signature_date'])

print(df.isnull().sum())

#encoding categorical and changing the datatype for year_ended
df = pd.get_dummies(df, columns=['HEADQUARTER_COUNTRY_CODE', 'auditor_home_office_state' ], drop_first=True) ##doubcheck, make sure the benchmark country is the same for both variables













