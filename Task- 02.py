#!/usr/bin/env python
# coding: utf-8

# PRODIGY INFOTECH

# Author: Mansi Jadhav

# Data Science Intern

# Task - 02: Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from the Kaggle. Explore the relationships between variables and identify patterns and trends in the data.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("titanic.csv")    #Loading the dataset


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.info()


# In[6]:


#Checking shape of the dataset
print("Rows: ", data.shape[0])
print("Columns: ", data.shape[1])


# In[7]:


#Checking number of unique values in all the columns
data.nunique()


# In[9]:


data.describe()


# In[10]:


#Checking for missing values
data.isnull()


# In[11]:


data.isnull().sum()


# In[12]:


#Filling missing values with the median age in Age column
data['Age'].fillna(data['Age'].median(), inplace= True)


# In[13]:


# Handling missing values in 'Cabin' column (create a binary column)
data['cabin_missing'] = data['Cabin'].isnull().astype(int)


# In[14]:


#Filling missing values with the mode of Embarked column
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)


# In[15]:


data.head()


# In[16]:


data.isnull().sum()


# In[17]:


data.dtypes


# In[21]:


#Pie plot for survived & non- survived percentage; 0- not survived, 1 - survived
Survived = data['Survived'].value_counts()  
plt.pie(
    x= Survived, 
    labels=Survived.index,
    autopct='%1.1f%%',
    startangle=90
)
_ = plt.title("'Survived' distribution")


# In[23]:


#Histogram for Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[24]:


#Bar plot for gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(data= data, x='Sex')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[27]:


#Box plot for Age vs Survival
plt.figure(figsize=(8, 6))
sns.boxplot(data= data, x='Survived', y='Age', palette=['#ccccff','#ffcc99'])
plt.title('Survival vs. Age')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()


# In[32]:


#Barplot for Survival vs Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Pclass', hue='Survived', palette = ['#0059b3','#ff66b3'])
plt.title('Survival vs. Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[42]:


#Countplot for Port of Embarkation
plt.figure(figsize=(8, 6))
sns.countplot(data= data, x='Embarked', hue='Survived', palette='Set2')
plt.title('Survival vs. Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()


# In[43]:


#Violin plot for Age and Gender
plt.figure(figsize=(8, 6))
sns.violinplot(data= data, x='Sex', y='Age', hue='Survived', palette='pastel')
plt.title('Age and Gender vs. Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()


# In[44]:


#Pair plot eith hue for Pclass vs Age
sns.pairplot(data, hue='Pclass')
plt.suptitle('Pair Plot with Hue for Class', y=1.02)
plt.show()


# In[45]:


#Box plot for fare distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data= data, y='Fare', color='lightblue')
plt.title('Fare Distribution')
plt.ylabel('Fare')
plt.show()


# In[46]:


#Swarm plot for Age & Gender
plt.figure(figsize=(8, 6))
sns.swarmplot(data= data, x='Sex', y='Age', hue='Survived', palette='Set1')
plt.title('Age and Gender vs. Survival (Swarm Plot)')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()


# In[39]:


numeric_columns = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()
correlation_matrix


# In[40]:


#Heatmap for numerical variables
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[51]:


#Heatmap for categorical data
categorical_data = data[['Sex','Ticket', 'Cabin', 'Embarked']]
categorical_data_encoded = categorical_data.apply(lambda x: x.astype('category').cat.codes)
sns.heatmap(categorical_data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Categorical Data')
plt.show()


# In[34]:


#Pair plot for multivariate analysis
sns.pairplot(data, hue='Survived')
plt.suptitle('Pair Plot', y=1.02)
plt.show()

