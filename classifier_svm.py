# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:15:36 2018

@author: lingyun
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split 
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("words.csv", header=None, encoding="utf-8")
#X = np.array(data.iloc[:,0:44])
#y = np.array(data.iloc[:,-1])
X = data.iloc[:,0:44]
X = StandardScaler().fit_transform(X)
y = data.iloc[:,-1]

#设置PCA的目标维数并创建一个model
pca = PCA(n_components=3).fit(X)
#取得目标向量
z = pca.fit_transform(X)
#取得前八十回的向量
xs_a = [row[0] for row in z[:80]]
ys_a = [row[1] for row in z[:80]]
zs_a = [row[2] for row in z[:80]]
#取得后四十回的向量
xs_b = [row[0] for row in z[-40:]]
ys_b = [row[1] for row in z[-40:]]
zs_b = [row[2] for row in z[-40:]]
 
#创建一个新的图表
fig = plt.figure()
#ax = fig.add_subplot(111)二维
ax = Axes3D(fig)
#绘图
ax.set_title('1-80 & 81-120 chapters\n') 
ax.scatter(xs_a, ys_a, zs_a, s=50, c='b', marker='o')
ax.scatter(xs_b, ys_b, zs_b, s=50, c='r', marker='o')
plt.legend(['1-80','81-120'])
plt.show()    
#########################
xs_a = [row[0] for row in z[:40]]
ys_a = [row[1] for row in z[:40]]
zs_a = [row[2] for row in z[:40]]

xs_b = [row[0] for row in z[40:80]]
ys_b = [row[1] for row in z[40:80]]
zs_b = [row[2] for row in z[40:80]]
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title(' 1-40 & 41-80 chapters\n') 
ax.scatter(xs_a, ys_a, zs_a, s=50, c='b', marker='o')
ax.scatter(xs_b, ys_b, zs_b, s=50, c='r', marker='o')
plt.legend(['1-40','41-80'])
plt.show()

#####################3
xs_a = [row[0] for row in z[:40]]
ys_a = [row[1] for row in z[:40]]
zs_a = [row[2] for row in z[:40]]

xs_b = [row[0] for row in z[40:80]]
ys_b = [row[1] for row in z[40:80]]
zs_b = [row[2] for row in z[40:80]]

xs_c = [row[0] for row in z[-40:]]
ys_c = [row[1] for row in z[-40:]]
zs_c = [row[2] for row in z[-40:]]
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('1-40 & 41-80 & 81-120 chapters\n') 
ax.scatter(xs_a, ys_a, zs_a, s=50, c='b', marker='o')
ax.scatter(xs_b, ys_b, zs_b, s=50, c='r', marker='o')
ax.scatter(xs_c, ys_c, zs_c, s=50, c='g', marker='o')
plt.legend(['1-40','41-80', '81-120'])
plt.show() 

''''''
# 分出训练和测试样本
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
# 使用GridSearch找到合适的参数
params = [{'C':[1,5,10,50,100,250,500]}]
#自动调整参数
grid = GridSearchCV(SVC(kernel='linear'),params,cv=10)
# 训练 预测
grid.fit(x_train,y_train)
y_pred = grid.predict(x_test)
# 显示主要的分类指标，返回每个类标签的精确率、召回率及F1值
print(classification_report(y_test, y_pred))



