#In[]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
def singularValue(data,boundary):
    data = np.array(data)
    mean = int(data.mean())
    for i in range(len(data)):
        if data[i] > boundary:
            data[i] = mean
    return data


data_train['卧室数量'] = singularValue(np.array(data_train['卧室数量']),5)
data_train['厅的数量'] = singularValue(np.array(data_train['厅的数量']),4)
data_train['卫的数量'] = singularValue(np.array(data_train['卫的数量']),3)

data_test['卧室数量'] = singularValue(np.array(data_test['卧室数量']),5)
data_test['厅的数量'] = singularValue(np.array(data_test['厅的数量']),4)
data_test['卫的数量'] = singularValue(np.array(data_test['卫的数量']),3)
data_test.drop("id", axis = 1, inplace = True)


for col in ['距离','地铁站点','地铁线路','居住状态']:
    data_train[col] = data_train[col].fillna(0)
    data_test[col] =data_test[col].fillna(0)
    
for col in ['小区房屋出租数量','位置','区','装修情况','出租方式']:
    data_train[col] = data_train[col].fillna(data_train[col].median())
    data_test[col] = data_test[col].fillna(data_test[col].median())

houseOrientation = pd.get_dummies(data_train[['房屋朝向']])
data_train = pd.merge(data_train, houseOrientation,left_index=True,right_index=True)
data_train= data_train.drop('房屋朝向',1)

houseOrientation = pd.get_dummies(data_test[['房屋朝向']])
data_test = pd.merge(data_test, houseOrientation,left_index=True,right_index=True)
data_test= data_test.drop('房屋朝向',1)

array = data_train.values
X = array[:, :-1]
Y = array[:,-1]
pca = PCA(n_components =12)
pca.fit(X)
X = pca.transform(X)
#data_test = pca.transform(data_test)
#data_test_y = np.zero((len()))
#tsne = TSNE(n_components=2)
#tsne.fit(X)
#X = tsne.fit_transform(X)
seed = 7
validation_size = 0.3
X,X_test,Y,Y_test = train_test_split(X,Y,test_size=validation_size,random_state=seed)
# rf = RandomForestClassifier()
# rfe = RFE(estimator=rf,n_features_to_select=7)
# X = rfe.fit_transform(X,Y)
# X_test = rfe.fit_transform(X_test,Y_test)


num_folds = 10
scoring = 'neg_mean_squared_error'

model = {}
model['LR'] = LinearRegression()
model['LASSO'] = Lasso()
model['EN'] = ElasticNet()
model['KNN'] = KNeighborsRegressor()
model['CART'] = DecisionTreeRegressor()
model['LGB'] = lgb.LGBMRegressor()

results = []
for key in model:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(model[key],X,Y,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s:%f (%f)'% (key,cv_result.mean(),cv_result.std()))

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
X_test = scaler.transform(X_test)
model_lgb = lgb.LGBMRegressor(learning_rate= 0.1 ,max_depth=-1 ,reg_alpha=0.01,
                              min_child_samples=40)
model_lgb.fit(X,Y)
predictions = model_lgb.predict(X_test)
print(mean_squared_error(Y_test,predictions))
print(model_lgb.feature_importances_)
print("hello")