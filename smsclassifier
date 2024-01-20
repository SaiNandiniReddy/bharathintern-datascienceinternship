#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the SMS dataset
# You can replace 'your_dataset.csv' with the actual file path or URL of your dataset
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
sms_data = pd.read_table(url, header=None, names=['label', 'message'])

# Display the first few rows of the dataset
print(sms_data.head())

# Convert labels to numerical values (0 for 'ham' and 1 for 'spam')
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_data['message'], sms_data['label'], test_size=0.2, random_state=42)

# Create a Bag-of-Words model (CountVectorizer)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create a Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Print classification report for more detailed metrics
print(classification_report(y_test, predictions))


# In[ ]:




