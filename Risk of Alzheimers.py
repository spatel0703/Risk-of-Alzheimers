#!/usr/bin/env python
# coding: utf-8

# # Data Quality report

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import zscore

# Load and preprocess data
df = pd.read_csv('oasis_longitudinal.csv', na_filter=False) 
df


# In[2]:


print(df.describe())


# In[3]:


#data quality report
#Missing values
missing_vals = df.isnull().sum()
print("Missing values:\n", missing_vals)


# In[4]:


# data quality report for categorical features
categorical_feat = ['Group', 'M/F', 'Hand']
categorical_data_qual = pd.DataFrame(index=categorical_feat, columns=['Count', 'Unique', 'Top', 'Freq'])

# This code will pull the count for each categorical feature and report the unique values
# Will also pull most occuring value for each feature and its frequency
for feature in categorical_feat:
    categorical_data_qual.loc[feature] = [df[feature].count(), df[feature].nunique(), df[feature].mode().iloc[0], df[feature].value_counts().max()]
print("Data Quality Report for Categorical Features:\n", categorical_data_qual)


# In[5]:


# data quality report for continuous features
continuous_feat = ['Visit', 'MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
continuous_data_qual = pd.DataFrame(index=continuous_feat, columns=['Count', 'Unique', 'Mode', 'Max Count'])

# This code will pull the count for each contiuous feature and report the unique values
# Will also pull most occuring value for each feature and its frequency
for feature in continuous_feat:
    continuous_data_qual.loc[feature] = [df[feature].count(), df[feature].nunique(), df[feature].mode().iloc[0], df[feature].value_counts().max()]

print("Data Quality Report for Continuous Features:\n", continuous_data_qual)


# # Data Visualization

# In[6]:


import math

# Determine the number of rows and columns based on the total number of features
total_features = len(categorical_feat) + len(continuous_feat)
rows = math.ceil(total_features / 3)  # Assuming 3 columns

# Set up the figure
plt.figure(figsize=(15, 4 * rows))

# Plot histograms for categorical features
for i, feature in enumerate(categorical_feat, 1):
    plt.subplot(rows, 3, i)
    sns.countplot(x=feature, data=df)
    plt.title(f'{feature} Distribution')

# Plot histograms for continuous features
for i, feature in enumerate(continuous_feat, 1):
    plt.subplot(rows, 3, i + len(categorical_feat))
    sns.histplot(df[feature], bins=20, kde=True)
    plt.title(f'{feature} Distribution')

# Adjust layout
plt.tight_layout()
plt.show()


# In[7]:


# Determine the number of rows and columns based on the total number of features
total_features = len(categorical_feat) + len(continuous_feat)
rows = math.ceil(total_features / 3)  # Assuming 3 columns

# Set up the figure
plt.figure(figsize=(15, 4 * rows))

# Boxplots for categorical features
for i, feature in enumerate(categorical_feat, 1):
    plt.subplot(rows, 3, i)
    sns.boxplot(x=feature, y='Age', data=df, color='blue')  # Adjust 'Age' as needed, and set color if desired
    plt.title(f'{feature} Boxplot')

# Boxplots for continuous features
for i, feature in enumerate(continuous_feat, 1):
    plt.subplot(rows, 3, i + len(categorical_feat))
    sns.boxplot(x=feature, y='Age', data=df, color='blue')  # Adjust 'Age' as needed, and set color if desired
    plt.title(f'{feature} Boxplot')

# Adjust layout
plt.tight_layout()
plt.show()


# # Normalization of Original Dataset

# In[8]:


from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

# Step 1: Handle missing values
normalized_df = df.copy()

# Convert non-numeric values in continuous features to numeric
normalized_df[continuous_feat] = normalized_df[continuous_feat].apply(pd.to_numeric, errors='coerce')

# Check for NaN values after conversion
print("NaN values after conversion:\n", normalized_df.isnull().sum())

# Fill NaN values with the mean

# Step 2: Min-Max Scaling (Normalization)
scaler = MinMaxScaler()
normalized_df[continuous_feat] = scaler.fit_transform(normalized_df[continuous_feat])

# Print updated statistics after normalization (without outlier removal)
print(normalized_df.describe())


# In[9]:


normalized_df
normalized_df.to_csv('normalized_results.csv', index=False)
 


# # Imputation and Outliers

# In[10]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Step 1: Handle missing values
imputed_df = df.copy()

# Convert non-numeric values in continuous features to numeric
imputed_df[continuous_feat] = imputed_df[continuous_feat].apply(pd.to_numeric, errors='coerce')

# Check for NaN values after conversion
print("NaN values after conversion:\n", imputed_df.isnull().sum())

# Impute missing values with the median
imputer = SimpleImputer(strategy='median')
imputed_df[continuous_feat] = imputer.fit_transform(imputed_df[continuous_feat])

# Step 2: Outlier Removal
outlier_threshold = 3  # Adjust this threshold based on your data distribution
imputed_df = imputed_df[(np.abs(zscore(imputed_df[continuous_feat])) < outlier_threshold).all(axis=1)]

# Rest of the code remains the same

# Print updated statistics after outlier removal
print(imputed_df.describe())


# In[11]:


imputed_df


# In[12]:


# Determine the number of rows and columns based on the total number of features
total_features = len(categorical_feat) + len(continuous_feat)
rows = math.ceil(total_features / 3)  # Assuming 3 columns

# Set up the figure
plt.figure(figsize=(15, 4 * rows))

# Plot histograms for categorical features
for i, feature in enumerate(categorical_feat, 1):
    plt.subplot(rows, 3, i)
    sns.countplot(x=feature, data=imputed_df)
    plt.title(f'{feature} after outlier removal')

# Plot histograms for continuous features
for i, feature in enumerate(continuous_feat, 1):
    plt.subplot(rows, 3, i + len(categorical_feat))
    sns.histplot(imputed_df[feature], bins=20, kde=True)
    plt.title(f'{feature} after outlier removal')

# Adjust layout
plt.tight_layout()
plt.show()


# # Accuracy Matrix and Random Forest Classifier

# In[13]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 3: Build Classifier
X_continuous = imputed_df[continuous_feat]
X_categorical = imputed_df[categorical_feat]

# One-hot encode categorical features
encoder = OneHotEncoder()
X_categorical_encoded = encoder.fit_transform(X_categorical).toarray()

# Concatenate continuous and encoded categorical features
X = np.concatenate((X_continuous, X_categorical_encoded), axis=1)

y = df.loc[imputed_df.index, 'Group']  # Target variable with consistent indices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = classifier.predict(X_test)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optionally, you can also check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")


# In[14]:


# Function to get user input for continuous and categorical features
def get_user_input():
    user_input = {}
    
    # Continuous features
    for feature in continuous_feat:
        user_input[feature] = float(input(f"{feature}: "))
    
    # Categorical features
    for feature in categorical_feat:
        user_input[feature] = input(f"{feature}: ")

    return user_input

# Get user input
user_input = get_user_input()

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Print User Input
print("\nUser Input:")
print(user_df.to_string(index=False))

# Handle missing values in user input for continuous features
user_df[continuous_feat] = user_df[continuous_feat].apply(pd.to_numeric, errors='coerce')
user_df[continuous_feat].fillna(user_df[continuous_feat].mean(), inplace=True)

# Print User DataFrame after handling missing values for continuous features
print("\nUser DataFrame after handling missing values for continuous features:")
print(user_df.to_string(index=False))

# Identify outliers in user input for continuous features
user_input_continuous = user_df[continuous_feat].apply(pd.to_numeric, errors='coerce')
user_input_continuous.fillna(user_input_continuous.mean(), inplace=True)

outliers_continuous = (np.abs(zscore(user_input_continuous)) >= outlier_threshold).any(axis=1)
user_df['Outlier'] = outliers_continuous

# Print User DataFrame with Outliers Identified for continuous features
print("\nUser DataFrame with Outliers Identified for continuous features:")
print(user_df.to_string(index=False))

# One-hot encode categorical features
user_categorical_encoded = encoder.transform(user_df[categorical_feat]).toarray()

# Concatenate continuous and encoded categorical features
user_input_combined = np.concatenate((user_df[continuous_feat], user_categorical_encoded), axis=1)

# Predict the probability of Alzheimer's
probability_alzheimer = classifier.predict_proba(user_input_combined)[:, 1]
print(f"\nProbability of Alzheimer's: {probability_alzheimer[0]:.2%}")


# # Adding Risk column to imputed and outlier dataset

# In[15]:


# One-hot encode categoricals on full dataset 
X_categorical_full = encoder.transform(imputed_df[categorical_feat]).toarray()  

# Concatenate with continuous columns
X_full = np.concatenate((imputed_df[continuous_feat], X_categorical_full), axis=1)

# Now make predictions
y_pred_proba = classifier.predict_proba(X_full)[:,1]

# Rest of code remains same
risk_labels = ['Low' if prob < 0.5 else 'High' for prob in y_pred_proba]
imputed_df['Risk of Alzheimer'] = risk_labels

imputed_df


# In[16]:


#This was the original code to add the risk column to the outliers dataset
outliers_df = pd.read_csv('outliers_result.csv', na_filter=False) 

# One-hot encode categoricals on outliers_df
X_categorical_outliers = encoder.transform(outliers_df[categorical_feat]).toarray()

# Concatenate with continuous columns
X_outliers = np.concatenate((outliers_df[continuous_feat], X_categorical_outliers), axis=1)

# Now make predictions
y_pred_proba_outliers = classifier.predict_proba(X_outliers)[:, 1]

# Rest of the code remains the same
risk_labels_outliers = ['Low' if prob < 0.5 else 'High' for prob in y_pred_proba_outliers]
outliers_df['Risk of Alzheimer'] = risk_labels_outliers

# Display the resulting DataFrame
outliers_df


# In[17]:


# Determine the number of rows and columns based on the total number of features
total_features = len(categorical_feat) + len(continuous_feat)
rows = math.ceil(total_features / 3)  # Assuming 3 columns

# Set up the figure
plt.figure(figsize=(15, 4 * rows))

# Plot histograms for categorical features
for i, feature in enumerate(categorical_feat, 1):
    plt.subplot(rows, 3, i)
    sns.countplot(x=feature, data=outliers_df)
    plt.title(f'{feature} Outlier Distribution')

# Plot histograms for continuous features
for i, feature in enumerate(continuous_feat, 1):
    plt.subplot(rows, 3, i + len(categorical_feat))
    sns.histplot(outliers_df[feature], bins=20, kde=True)
    plt.title(f'{feature} Outlier Distribution')

# Adjust layout
plt.tight_layout()
plt.show()


# # Data Visualization for imputed and outlier datasets

# In[18]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# One-hot encode categoricals on full dataset  
X_categorical_full = encoder.transform(imputed_df[categorical_feat]).toarray()

# Concatenate continuous columns  
X_full = np.concatenate((imputed_df[continuous_feat], X_categorical_full), axis=1)

y_true = imputed_df['Group']
y_true = [1 if each == 'Demented' else 0 for each in y_true] 

# Get predicted probabilities for class 1 (Alzheimer's)
y_prob = classifier.predict_proba(X_full)[:, 1]  

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# In[19]:


plt.figure(figsize=(15, 10))

for i, feat in enumerate(continuous_feat):
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=feat, y='Age', data=imputed_df, hue='Risk of Alzheimer', palette='viridis')
    plt.title(f'Scatter Plot of {feat} vs Age')
    plt.xlabel(feat)
    plt.ylabel('Age')
    plt.legend(title='Risk of Alzheimer', loc='upper right')

plt.tight_layout()
plt.show()


# In[20]:


from matplotlib.patches import Patch

colors = ['green', 'blue']

plt.figure(figsize=(15, 10))

for i, feat in enumerate(continuous_feat):
    plt.subplot(3, 4, i + 1)
    
    for j, risk_level in enumerate(imputed_df['Risk of Alzheimer'].unique()):
        subset = imputed_df[imputed_df['Risk of Alzheimer'] == risk_level]
        sns.histplot(x=feat, data=subset, color=colors[j], label=risk_level, bins=30, alpha=0.7)

    plt.title(f'Histogram of {feat} by Risk of Alzheimer')
    plt.xlabel(feat)
    plt.ylabel('Count')
    
    # Create a custom legend for each subplot
    if i == 0:
        plt.legend(handles=[Patch(color=color, label=risk_level) for color, risk_level in zip(colors, imputed_df['Risk of Alzheimer'].unique())], title='Risk of Alzheimer', loc='upper right')
    else:
        plt.legend().set_visible(False)

plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(15, 5))

for i, feat in enumerate(categorical_feat):
    plt.subplot(1, 3, i + 1)
    
    for j, risk_level in enumerate(imputed_df['Risk of Alzheimer'].unique()):
        subset = imputed_df[imputed_df['Risk of Alzheimer'] == risk_level]
        sns.histplot(x=feat, data=subset, color=colors[j], label=risk_level, alpha=0.7, discrete=True, multiple='stack')

    plt.title(f'Histogram of {feat} by Risk of Alzheimer')
    plt.xlabel(feat)
    plt.ylabel('Count')
    
    # Create a custom legend for each subplot
    if i == 0:
        plt.legend(handles=[Patch(color=color, label=risk_level) for color, risk_level in zip(colors, imputed_df['Risk of Alzheimer'].unique())], title='Risk of Alzheimer', loc='upper right')
    else:
        plt.legend().set_visible(False)

plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(15, 5))

for i, feat in enumerate(categorical_feat):
    plt.subplot(1, 3, i + 1)
    
    for j, risk_level in enumerate(outliers_df['Risk of Alzheimer'].unique()):
        subset = outliers_df[outliers_df['Risk of Alzheimer'] == risk_level]
        sns.histplot(x=feat, data=subset, color=colors[j], label=risk_level, alpha=0.7, discrete=True, multiple='stack')

    plt.title(f'Histogram of {feat} by Risk of Alzheimer (Outliers)')
    plt.xlabel(feat)
    plt.ylabel('Count')
    
    # Create a custom legend for each subplot
    if i == 0:
        plt.legend(handles=[Patch(color=color, label=risk_level) for color, risk_level in zip(colors, imputed_df['Risk of Alzheimer'].unique())], title='Risk of Alzheimer', loc='upper right')
    else:
        plt.legend().set_visible(False)

plt.tight_layout()
plt.show()


# In[23]:


# Set the style of seaborn for better visualization
sns.set(style="whitegrid")

# Plot heatmap for imputed_df
plt.figure(figsize=(12, 8))
sns.heatmap(imputed_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap - Imputed DataFrame")
plt.show()

# Plot heatmap for outliers_df
plt.figure(figsize=(12, 8))
sns.heatmap(outliers_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap - Outliers DataFrame")
plt.show()


# In[ ]:




