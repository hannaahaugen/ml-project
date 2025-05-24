from pydoc import describe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

#Load data and inspect
filename = 'dataset_MLproject.xlsx'
df= pd.read_excel('dataset_MLproject.xlsx')
df = df.drop(columns=['auditor_name'])
df = df.drop(columns=['AUDIT_OPINION_TYPE'])
df = df.drop(columns=['opinion_fiscal_year_end'])     


#TARGET VALUE BEFORE CLEANING
print(df["HAS_GOING_CONCERN_MODIFICATION"].unique())
print(df["HAS_GOING_CONCERN_MODIFICATION"].dtype)
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

df['IPO_YEAR'] = pd.to_datetime(df['IPO_DATE'], errors='coerce').dt.year
df['IPO_YEAR'] = df['IPO_YEAR'].fillna(-1)
df = df.drop(columns=['IPO_DATE'])

imputer = KNNImputer(n_neighbors=5)
df['TOTAL_EMPLOYEES'] = imputer.fit_transform(df[['TOTAL_EMPLOYEES']])

df = df.dropna(subset=['SIC_CODE_FKEY'])

counts = df['IS_EU_REGULATED_SUBMARKET'].value_counts()
print(counts)
df['IS_EU_REGULATED_SUBMARKET'] = df['IS_EU_REGULATED_SUBMARKET'].fillna(0)

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_MARKET_CAP_USD'] = imputer.fit_transform(df[['MOST_RECENT_MARKET_CAP_USD']])

imputer = KNNImputer(n_neighbors=5)
df['REVENUE_USD'] = imputer.fit_transform(df[['REVENUE_USD']])

imputer = KNNImputer(n_neighbors=5)
df['NET_INCOME_USD'] = imputer.fit_transform(df[['NET_INCOME_USD']])

imputer = KNNImputer(n_neighbors=5)
df['TOTAL_ASSETS_USD'] = imputer.fit_transform(df[['TOTAL_ASSETS_USD']])

imputer = KNNImputer(n_neighbors=5)
df['AUDIT_FEES_USD'] = imputer.fit_transform(df[['AUDIT_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['AUDIT_RELATED_FEES_USD'] = imputer.fit_transform(df[['AUDIT_RELATED_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_TAX_FEES_USD'] = imputer.fit_transform(df[['MOST_RECENT_TAX_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_OTHER_FEES_USD'] = imputer.fit_transform(df[['MOST_RECENT_OTHER_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['MOST_RECENT_NON_AUDIT_FEES_USD'] = imputer.fit_transform(df[['MOST_RECENT_NON_AUDIT_FEES_USD']])

imputer = KNNImputer(n_neighbors=5)
df['TOTAL_FEES_USD'] = imputer.fit_transform(df[['TOTAL_FEES_USD']])

df = df.dropna(subset=['auditor_home_office_state'])

df = df.dropna(subset=['opinion_signature_date'])
df['opinion_signature_date'] = pd.to_datetime(df['opinion_signature_date'], errors='coerce').dt.year
df = df.drop(columns=['opinion_signature_date'])

df = df.dropna(subset=['year_ended'])
df['year_ended'] = df['year_ended'].astype('int64')

print(df.isnull().sum())

# SIC inspection
counts = df['SIC_CODE_FKEY'].value_counts()
print(counts)

# Define SIC ranges
sic_ranges = {
    "Agriculture, Forestry and Fishing": (100, 999),
    "Mining": (1000, 1499),
    "Construction": (1500, 1799),
    "Not Used": (1800, 1999),
    "Manufacturing": (2000, 3999),
    "Transportation, Communications, Electric, Gas and Sanitary service": (4000, 4999),
    "Wholesale Trade": (5000, 5199),
    "Retail Trade": (5200, 5999),
    "Finance, Insurance and Real estate": (6000, 6799),
    "Services": (7000, 8999),
    "Public administration": (9100, 9729),
    "Nonclassifiable": (9900, 9999)
}

# Function to map SIC codes to their divisions
def map_sic_to_division(sic_code):
    for division, (start, end) in sic_ranges.items():
        if start <= sic_code <= end:
            return division
    return "Unknown"  # For codes outside the defined ranges

# Apply the mapping function
df["SIC_Division"] = df["SIC_CODE_FKEY"].apply(map_sic_to_division)

# Categorical features - encoding
categorical_features = ['HEADQUARTER_COUNTRY_CODE', 'auditor_home_office_state',
                        'AUDIT_OPINION_TYPE_FKEY', 'SIC_Division']
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(df[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns and add the encoded ones
df = df.drop(columns=categorical_features)
df = pd.concat([df, encoded_categorical_df], axis=1)

# Continuous features - scaling
continuous_features = [
    'TOTAL_EMPLOYEES', 'MOST_RECENT_MARKET_CAP_USD', 'REVENUE_USD', 'NET_INCOME_USD', 
    'TOTAL_ASSETS_USD', 'AUDIT_FEES_USD', 'AUDIT_RELATED_FEES_USD', 'MOST_RECENT_TAX_FEES_USD', 
    'MOST_RECENT_OTHER_FEES_USD', 'MOST_RECENT_NON_AUDIT_FEES_USD', 'TOTAL_FEES_USD', 
    'NUMBER_OF_AUDITORS'
]
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

##SETTING UP FOR MODELS (probably better to first export a dataframe that was cleaned and edited so no need to load the code every time)##
print(df.head())
col = df.pop("HAS_GOING_CONCERN_MODIFICATION")
df.insert(len(df.columns), "HAS_GOING_CONCERN_MODIFICATION", col)

dataset = df.to_numpy()
X = dataset[:, :-1]  # the features
Y = dataset[:, -1]  # the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)



