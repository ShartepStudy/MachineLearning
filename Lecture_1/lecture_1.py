import numpy as np
import pandas as pd
data = pd.read_csv('titanic.csv', index_col='PassengerId')
data.head()

# 1
sex_count = data['Sex'].value_counts()
print('Question 1:')
print(sex_count)
print('\n')

# 2
total_passengers = len(data)
total_survived = data['Survived'].value_counts()[1]
survived_percentage = total_survived / total_passengers * 100
print('Question 2:')
print(survived_percentage)
print('\n')

# 3
total_first_class = data['Pclass'].value_counts()[1]
first_class_percentage = total_first_class / total_passengers * 100
print('Question 3:')
print(first_class_percentage)
print('\n')

# 4
age_average = data['Age'].mean(axis=0)
age_mediane = data['Age'].median(axis=0)
print('Question 4:')
print(age_average)
print(age_mediane)
print('\n')

# 5
pearson_correlation = data[['SibSp', 'Parch']].corr()
print('Question 5:')
print(pearson_correlation)
print('\n')

# 6
first_names = data['Name'].apply(lambda x : pd.Series(x.split(' ')))
first_names[0].value_counts()
