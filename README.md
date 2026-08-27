# Feature-engineering-for-titatnic-dataset
Feature engineering using different encoding techniques for titatnic dataset
# Titanic Data Cleaning & Machine Learning

**Author:** Eyoba Mulubrhan
**Project Goal:** Learn how to clean data, fix missing values, and use two different categorical encoding methods (Label Encoding and One-Hot Encoding) to prepare data for a Logistic Regression model.


## Project Datasets
Make sure you upload these three files to your folder before running the notebook:
* **`titanic.csv`**: The main Titanic dataset with passenger records.
* **`titanic_ages.csv`**: A list used to look up missing age records.
* **`titanic_embarked.csv`**: A list used to look up missing embarkation ports.

## Repository Structure
* **`Feature engineering.ipynb`**: The main Jupyter Notebook where all the code is written.


## 1. Library Imports & Data Loading
* Imported basic Python libraries (`pandas`, `numpy`, `matplotlib`).
* Loaded the primary Titanic dataset and inspected the first 5 rows.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_titanic = pd.read_csv('titanic.csv')
df_titanic.head(5)
```

## 2. Initial Data Cleaning

### Precise Missing Value Imputation
Instead of filling missing data with global averages, I used exact lookup information from separate files:
* **Age Lookup**: Filled missing values by matching passenger names with verified ages in `titanic_ages.csv`.

```python
age_lookup = a_df.set_index('Name')['Age']
df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Name'].map(age_lookup))
```

### Targeted Manual Corrections
I ran `df_titanic.info()` and checked for leftover missing entries. A few rows still had null fields, so I targeted them manually using string filters:
* Found "Peduzzi, Mr. Joseph" and set his missing age to `24`.
* Found "Icard, Miss. Amelie" and "Stone, Mrs. George Nelson" and set their missing embarkation ports to `'S'` (Southampton).

```python
# Fix remaining specific passenger data manually
mask = df_titanic['Name'].str.contains('Peduzzi, Mr. Joseph', na=False)
df_titanic.loc[mask, 'Age'] = 24

mask1 = df_titanic['Name'].str.contains('Icard, Miss. Amelie', na=False)
df_titanic.loc[mask1, 'Embarked'] = 'S'
```

### Separating Features and Target
* **Drop Cabin**: Removed the `Cabin` column entirely because it had too many missing values.
* **Isolate Target**: Separated the features (`x`) from our prediction target (`y` = `Survived`).

```python
x = df_titanic.drop(['Cabin', 'Survived'], axis=1)
y = df_titanic['Survived']
```

### Handling Leftover Column Gaps
To make sure absolutely no null values remained, I filled any final gaps using basic column metrics:
* Filled remaining empty cells in `Age` and `Fare` with column **medians**.
* Filled remaining empty cells in `Embarked` with the most frequent value (**mode**).

---

## 3. Data Exploration
* Used a bar chart to plot the value counts of our target variable (`y`). This let me check for imbalances between survivors (1) and non-survivors (0).


## 4. Modeling Setup
I created a baseline modeling pipeline using a custom function named `logistic(x, y)`. This function automatically:
1. Splits the data into **80% training** and **20% testing** sets.
2. Trains a Scikit-Learn `LogisticRegression` classifier.
3. Prints out the final model accuracy.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logistic(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=44, test_size=0.2)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print('Accuracy : ', accuracy_score(y_test, y_pred))
```


## 5. Experiment 1: Label Encoding

First, I tested **Label Encoding** to convert categorical text columns into numbers by giving each category an integer value.

* **Implementation**: Cycled through columns and transformed `object` columns into integers using `LabelEncoder`.
* **Result**: Running `logistic(train, y)` yielded a baseline accuracy of **`81.68%`**. It also triggered a `ConvergenceWarning`, showing that the model struggled to optimize.

```python
from sklearn.preprocessing import LabelEncoder
# Looped over columns to turn objects into numeric label classes
```


## 6. Experiment 2: One-Hot Encoding

Next, I tested **One-Hot Encoding** to see if turning categories into separate binary columns (0 or 1) would yield better performance.

* **The Problem**: Using simple tools like `pd.get_dummies` creates over **2,250 columns** because every passenger has a unique name.
* **The Solution**: I wrote a custom loop using Scikit-Learn's `OneHotEncoder` to selectively encode text fields into separate dense blocks, then glued them side-by-side with numerical features.

```python
from sklearn.preprocessing import OneHotEncoder

df_list = []
label = OneHotEncoder(sparse_output=False)

for c in x.columns:
    if x[c].dtype == 'object':
        encoded_array = label.fit_transform(x[[c]])
        
        col_names = label.get_feature_names_out([c])
        temp_df = pd.DataFrame(encoded_array, columns=col_names, index=x.index)
        df_list.append(temp_df)
    else:
        df_list.append(x[[c]])

train2 = pd.concat(df_list, axis=1)
```

* **Result**: Running `logistic(train2, y)` raised the classification score to **`85.49%`**. 


