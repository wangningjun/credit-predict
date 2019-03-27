
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
#注意数据分割的几种方法
'''
ss=StratifiedShuffleSplit(n_splits=5,test_size=0.25,train_size=0.75,random_state=0)#分成5组，测试比例为0.25，训练比例是0.75
for train_index, test_index in ss.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)#获得索引值
   X_train, X_test = X[train_index], X[test_index]#训练集对应的值
   y_train, y_test = y[train_index], y[test_index]#类别集对应的值
'''
load_train = "credit_risk_train.csv"
load_test = "credit_risk_test.csv"

def data_process(load_train,load_test):

    # 读取数据，设置需要处理的列头
    df = pd.read_csv(load_train)
    df_test = pd.read_csv(load_test)
    feature_index = ["BILL_1","BILL_2","BILL_3","BILL_4","BILL_5","BILL_6","AGE","SEX_le","EDUCATION_le","MARRIAGE_le"]
    encoder_index = ["SEX", "EDUCATION", "MARRIAGE", "RISK"]
    stande_index  = ["BILL_1","BILL_2","BILL_3","BILL_4","BILL_5","BILL_6"]

    # 数据标准化
    for i in stande_index:
        df_s = StandardScaler().fit_transform(df[i][:, np.newaxis])
        df[i+"_s"] = df_s
    df_s = df.drop(stande_index, axis=1)
    # 对每一个字符元素做LabelEncoder处理，并添加
    for i in encoder_index:
        df_encoder = LabelEncoder().fit_transform(df_s[i])
        df_s[i+"_le"] = df_encoder
    # 删除旧有的列元素
    df_new = df_s.drop(encoder_index, axis=1)
    # 分割数据集
    train_features, test_features, train_labels, test_labels = train_test_split(
        df_new[feature_index], df_new['RISK_le'], test_size=0.2, random_state=42)
    return train_features, test_features, train_labels, test_labels

def svm_c(x_train, x_test, y_train, y_test):
     # rbf核函数，设置数据权重
     svc = SVC(kernel='rbf', class_weight='balanced',)
     c_range = np.logspace(-5, 10, 3, base=2)
     gamma_range = np.logspace(3, 5, 3, base=2)
     # 网格搜索交叉验证的参数范围，cv=3,3折交叉
     param_grid = [{'kernel': ['rbf'], 'C':c_range, 'gamma': gamma_range}]
     grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
     # 训练模型
     clf = grid.fit(x_train, y_train)
     # 计算测试集精度
     print(clf.best_params_)
     score = clf.score(x_test, y_test)
     print('精度为%s' % score)

def svm(x_train, x_test, y_train, y_test):
    svc = SVC(kernel='sigmoid', class_weight="balanced", C=5.6, gamma=8)
    clf = svc.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print('精度为%s' % score)

def rf(x_train, x_test, y_train, y_test):
    param_grid1 = {'max_features':range(3,11,2)}
    Rf = RandomForestClassifier(n_estimators=70, max_depth=10, min_samples_leaf=10,min_samples_split=50, random_state=0)
    grid = GridSearchCV(Rf, param_grid1)
    clf = grid.fit(x_train, y_train)
    print(clf.best_params_, clf.score(x_test, y_test))

    # clf = Rf.fit(x_train, y_train)
    # scores = clf.score(x_test, y_test)
    # print(scores)

if __name__ == '__main__':
    train_features, test_features, train_labels, test_labels=data_process(load_train, load_test)
    # svm_c(train_features, test_features, train_labels, test_labels)
    rf(train_features, test_features, train_labels, test_labels)


'''
{'max_depth': 13, 'min_samples_split': 50} 0.775
{n_estimators=70, max_depth=10, min_samples_split=50} 0.77525
'''