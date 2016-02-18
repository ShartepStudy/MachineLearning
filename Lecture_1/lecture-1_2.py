import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')
# remove unused columns
data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1)
# remove rows with NaN's values
data = data.dropna()

data.loc[data['Sex'] == 'male', 'Sex'] = True
data.loc[data['Sex'] == 'female', 'Sex'] = False

# split DataFrame into 2 parts (one for learning, one for testing)
learn = data[:700]
test = data[700:]


y = np.array(learn['Survived'])
test_answer = np.array(test['Survived'])

# remove unused columns
learn.drop(['Survived'], inplace=True, axis=1)
test.drop(['Survived'], inplace=True, axis=1)

X = np.array(learn)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_

print importances
print

predict = clf.predict(test)

errors = 0
for i in range(0, len(predict)):
    if test_answer[i] != predict[i]:
        errors += 1

print "total rows: "
print len(data)
print

print "learning rows:"
print len(learn)
print

print "test rows:"
print len(test)
print

print "Errors count:"
print errors
print

print "Errors %:"
print (errors * 100) / len(test)
print