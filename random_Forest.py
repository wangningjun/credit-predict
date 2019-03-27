from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def data_process(load_train,load_test):

    # 读取数据，设置需要处理的列头
    df = pd.read_csv(load_train)
    df_test = pd.read_csv(load_test)
    feature_index = ["BILL_1","BILL_2","BILL_3","BILL_4","BILL_5","BILL_6","AGE","SEX_le","EDUCATION_le","MARRIAGE_le"]
    encoder_index = ["SEX", "EDUCATION", "MARRIAGE", "RISK"]
    stande_index  = ["BILL_1","BILL_2","BILL_3","BILL_4","BILL_5","BILL_6"]

    # 数据标准化
    '''
    for i in stande_index:
        df_s = StandardScaler().fit_transform(df[i][:, np.newaxis])
        df[i+"_s"] = df_s
    df_s = df.drop(stande_index, axis=1)
    '''
    # 对每一个字符元素做LabelEncoder处理，并添加
    for i in encoder_index:
        df_encoder = LabelEncoder().fit_transform(df[i])
        df[i+"_le"] = df_encoder
    # 删除旧有的列元素
    df_new = df.drop(encoder_index, axis=1)
    # 分割数据集
    train_features, test_features, train_labels, test_labels = train_test_split(
        df_new[feature_index], df_new['RISK_le'], test_size=0.2, random_state=42)
    return train_features, test_features, train_labels, test_labels

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