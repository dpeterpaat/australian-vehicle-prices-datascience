import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv("avp.csv", na_values=['POA', '-', '- / -','Other', 'DEMO'],)

data = data.filter(['Brand','Kilometres', 'Price', 'UsedOrNew', 'CylindersinEngine'])
# data.isnull().sum()
data = data.dropna()

print(data.dtypes)

data = data.head(100)



# Check for first 5 & datatype

# print(data.dtypes)


# label_encoder = LabelEncoder()
# data['Price'] = label_encoder.fit_transform(data['Price'])
# data['Location'] = label_encoder.fit_transform(data['Location'])
data.drop(data[data['CylindersinEngine'] == "8 cyl"].index, inplace = True)

x = data['Kilometres']
y = data['Price']
c = data['CylindersinEngine']


classes = []


for v in c:
    if v == "4 cyl": 
        classes.append(1)
    elif v == "6 cyl":
        classes.append(0)
    else :
        classes.append(0)


print(len(classes))



newData = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(newData, classes)



plt.scatter(x, y,c=classes)
plt.show()

# plt.scatter(x,y)
# plt.show()

# x = np.array(data['Kilometres', 'Location'])
# y = np.array(data['Price'])

# x_train, x_classes, y_train, y_classes = train_test_split(x ,y, test_size=0.2)


# acc = knn.score(x_classes, y_classes)
# print("classes Accuracy:", acc)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

y_pred = knn.predict(xtest)

accuracy = accuracy_score(ytest,y_pred)

# accuracy = knn.score(x,y)

print('acc :' + accuracy)





# pred = clf.predict(x_classes)
# acc = accuracy_score(y_classes, pred)
# print(f"classes Accuracy: {acc}")



# accuracy = clf.score(x_classes, y_classes)
# print(accuracy)


# x = np.array(data['Kilometres'])
# y = np.array(data['Price'])

# xnew = x[0:100].astype(int).reshape(-1,1)
# ynew = y[0:100].astype(int).reshape(-1,1)

# xtrain, xclasses, ytrain, yclasses = train_classes_split(xnew, ynew, classes_size=0.5)

# model = LinearRegression()
# model.fit(xtrain, ytrain)

# accuracy = model.score(xclasses,yclasses)

# print(model.predict(np.array([40000]).reshape(-1,1)))
# print(accuracy)


# plt.scatter(xnew,ynew)
# plt.xlabel('Price')
# plt.ylabel('Kilometres')
# plt.plot(np.linspace(0,120000,6).reshape(-1,1), model.predict(np.linspace(0,300000,6).reshape(-1,1)), 'r')
# plt.show()



