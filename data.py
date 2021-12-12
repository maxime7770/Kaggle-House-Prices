import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')


print(train)
y_train = np.array(train['SalePrice'])
x_train = train.iloc[:, 1:-1].values

test = pd.read_csv('test.csv')
x_test = test.iloc[:, 1:-1].values

# which data are categorical? (ie) check for strings

cat = []
for col in train.columns:
    i = 0
    while pd.isna(train[col][i]):
        i += 1
    if isinstance(train[col][i], str):
        cat.append(col)

print(cat)


# plt.scatter(train.iloc[:, 17].values, train['SalePrice'])
# plt.xlabel(train.columns[17])
# plt.show()
