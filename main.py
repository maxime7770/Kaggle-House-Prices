import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression

train = pd.read_csv('train.csv')
train = train.fillna(0)
y_train = np.array(train['SalePrice'])
x_train = train.iloc[:, 1:-1].values
print(x_train)

test = pd.read_csv('test.csv')
test = test.fillna(0)
x_test = test.iloc[:, 1:-1].values
print(x_test)

train['Set'] = "Train"
test['Set'] = "Test"
DATA = train.append(test)
DATA.reset_index(inplace=True)

DATA['MSZoning'].fillna("RL", inplace=True)
DATA.LotFrontage.fillna(0, inplace=True)
DATA.Alley.fillna("NO", inplace=True)
DATA.Utilities.fillna('AllPub', inplace=True)
DATA.Exterior1st.fillna("VinylSd", inplace=True)
DATA.Exterior2nd.fillna("VinylSd", inplace=True)
DATA.MasVnrArea.fillna(0., inplace=True)
DATA.BsmtCond.fillna("No", inplace=True)
DATA.BsmtExposure.fillna("NB", inplace=True)
DATA.BsmtFinType1.fillna("NB", inplace=True)
DATA.BsmtFinSF1.fillna(0., inplace=True)
DATA.BsmtFinSF2.fillna(0., inplace=True)
DATA.BsmtUnfSF.fillna(0., inplace=True)
DATA.TotalBsmtSF.fillna(0., inplace=True)
DATA.Electrical.fillna("SBrkr", inplace=True)
DATA.BsmtFullBath.fillna(0., inplace=True)
DATA.BsmtHalfBath.fillna(0., inplace=True)
DATA.KitchenQual.fillna("TA", inplace=True)
DATA.Functional.fillna('Typ', inplace=True)
DATA.FireplaceQu.fillna("No", inplace=True)
DATA.GarageType.fillna("No", inplace=True)
DATA.GarageYrBlt.fillna(0, inplace=True)
DATA.GarageFinish.fillna("No", inplace=True)
DATA.GarageCars.fillna(0, inplace=True)
DATA.GarageArea.fillna(0, inplace=True)
DATA.GarageQual.fillna("No", inplace=True)
DATA.GarageCond.fillna("No", inplace=True)
DATA.PoolQC.fillna("No", inplace=True)
DATA.Fence.fillna("No", inplace=True)
DATA.MiscFeature.fillna("No", inplace=True)
DATA.SaleType.fillna("Con", inplace=True)
DATA.SaleCondition.fillna("Normal", inplace=True)


# categorical = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#                'BldgType', 'HouseStyle', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#                'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
#                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'OverallQual', 'LotConfig', 'YearBuilt', 'YearRemodAdd']
categorical = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


DATA = pd.get_dummies(DATA, columns=categorical)

TRAIN = DATA[DATA.Set == 'Train']
TEST = DATA[DATA.Set == 'Test']
HouseIds = TEST.Id.to_list()

TEST = TEST.drop(['Id', 'Set', "SalePrice", 'index'], axis=1)

y = TRAIN.SalePrice
X = TRAIN.drop(['SalePrice', 'Id', 'Set', 'index'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=13)

ridge_model = Ridge(alpha=10, random_state=13).fit(X_train, y_train)


print("train_score:", ridge_model.score(X_train, y_train))
print("test_score:", ridge_model.score(X_test, y_test))


y_predict_ridge = ridge_model.predict(TEST)
result = pd.DataFrame({"Id": HouseIds, "SalePrice": y_predict_ridge})
result.to_csv('submission.csv', index=False)


elasticnet = ElasticNet(alpha=1, random_state=10).fit(X_train, y_train)

y_predict_elastic = elasticnet.predict(TEST)
result2 = pd.DataFrame({"Id": HouseIds, "SalePrice": y_predict_elastic})
result2.to_csv('submission2.csv', index=False)
 
