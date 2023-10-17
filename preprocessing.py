# Import necessary modules
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (please replace the path with your own)
df = pd.read_csv("F:/MIT/NM assn/Data Analytics Using Cognos/Dataset/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Initial data exploration
print("Initial dataset shape:", df.shape)
df.info()

# Data cleaning: Remove duplicates and null values
df = df.drop_duplicates()
df = df.dropna()

# Verify removal of duplicates and null values
print("Count of missing values after cleaning:")
print(df.isnull().sum())
print("Count of duplicate values after cleaning:", df.duplicated().sum())

# Remove unnecessary columns (e.g., 'customerID' not related to churn)
df = df.drop('customerID', axis=1)
print("Dataset shape after dropping 'customerID':", df.shape)

# Replacing certain categorical values with binary values
# InternetService: DSL = 1, FiberOptic = 1, No = 0
df['FiberOptics'] = df.loc[:, 'InternetService']

internet_service_data = {'DSL': 1, 'Fiber optic': 1, 'No': 0}
df.replace({'InternetService': internet_service_data}, inplace=True)

# Further categorizing InternetService: DSL = 0, Fiber Optic = 1, No = 0
fiber_optic_data = {'DSL': 0, 'Fiber optic': 1, 'No': 0}
df.replace({'FiberOptics': fiber_optic_data}, inplace=True)

# Handling Internet-related data columns
internet_related_data = {'No': 0, 'Yes': 1, 'No internet service': 0}
internet_related_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df[internet_related_cols] = df[internet_related_cols].replace({k: v for k, v in internet_related_data.items()})

# Convert other columns to binary values
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0})

# Convert 'gender' to binary values: Male = 1, Female = 0
df['Gender_BIN'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Replacing binary values for 'MultipleLines'
phone_service = {'No phone service': 0, 'No': 0, 'Yes': 1}
df.replace({'MultipleLines': phone_service}, inplace=True)

# One-hot encoding for 'Contract' and 'PaymentMethod'
ohe = OneHotEncoder()
encoded_data = pd.DataFrame(ohe.fit_transform(df[['Contract', 'PaymentMethod']]).toarray())
encoded_data.columns = ohe.get_feature_names_out()
df = df.join(encoded_data)
df.drop(['Contract', 'PaymentMethod'], axis=1, inplace=True)

# Data preprocessing: scaling the non-binary columns
# Remove rows with empty 'TotalCharges' values and convert to float
empty = df[df['TotalCharges'] == " "].index 
df.drop(empty, inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Scaling columns: 'tenure', 'MonthlyCharges', and 'TotalCharges'
scalable_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
mm_scaler = MinMaxScaler()
df_scaling = pd.DataFrame(mm_scaler.fit_transform(df[scalable_columns]))
df_scaling.columns = mm_scaler.get_feature_names_out()

# Replace original columns with scaled versions
df.drop(scalable_columns, axis=1, inplace=True)
df = df.join(df_scaling)

# Ensure 'Churn' is the last column
df['Churn'] = df.pop('Churn')

# Final data exploration
print("Final dataset info:")
print(df.info())

# Save the cleaned and processed dataset to a new CSV file (please replace the path)
df.to_csv('F:/MIT/NM assn/Data Analytics Using Cognos/ output.csv', index=False)
