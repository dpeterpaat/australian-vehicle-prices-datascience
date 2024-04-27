import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



data = pd.read_csv("/Users/dicsypeter/Documents/Code/dataviz/avp.csv", )

x = np.array(data['Kilometres'])
y = np.array(data['Price'])

xnew = x[0:100].astype(int).reshape(-1,1)
ynew = y[0:100].astype(int).reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(xnew, ynew, test_size=0.5)

model = LinearRegression()
model.fit(xtrain, ytrain)

accuracy = model.score(xtest,ytest)

print(model.predict(np.array([40000]).reshape(-1,1)))
print(accuracy)


plt.scatter(xnew,ynew)
plt.xlabel('Price')
plt.ylabel('Kilometres')
plt.plot(np.linspace(0,120000,6).reshape(-1,1), model.predict(np.linspace(0,300000,6).reshape(-1,1)), 'r')
plt.show()


# df = pd.DataFrame(data)


# df = df.replace([np.nan], 0)

# x = df['Price'].astype(str)
# y = df['Kilometres'].astype(str)


# ]



# x10 = x[0:10]
# y10 = y[0:10]


# plt.scatter(xnew,ynew)
# # plt.ylim(0,100000)
# plt.show()



# print(coba['Year'])

#membersihkan data tahun yang kosong dan harga yang kosong
# data_filtered = data[(data['Year'] != '') & (data['Price'] != 0)]   

# data_tahun = data_filtered['Year']

# data2022 = data_filtered.loc[data_filtered['Year'] == 2022]

# print(data2022)

# y_data = coba['Location']

# print(x_data)

# plt.plot(x_data, x_data)


# plt.scatter(x_data,y_data)
# plt.show()

# print(data_tahun)

# from sklearn.datasets import load_breast_cancer

# data = load_breast_cancer(as_frame=True)

# print(data.frame)

