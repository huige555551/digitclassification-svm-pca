import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
# digits = datasets.load_digits()

digits = fetch_mldata('MNIST original')
# output dateset sample numbers and feature numbers
print(digits.data.shape)
# output target
print(np.unique(digits.target))
# output dataset
print(digits.data)



def train_for_model(digits):
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1, random_state=42)
    print("train", x_train.shape)
    print("test", x_test.shape)

    best = {'score':-1,'n_components':-1,'C':-1,'gamma':-1}
    for n_components in range(20, 40):
        pca_model = PCA(n_components=n_components, whiten=True)

        pca_model.fit(x_train)
        new_x_train = pca_model.transform(x_train)
        new_x_test = pca_model.transform(x_test)
        x_train = scale(x_train)
        x_test = scale(x_test)
        print('pca-n-components:{}'.format(n_components))

        for C in range(1, 5):
            for gamma in [1.0/new_x_train.shape[1], 0.1, 1]:
                svm_model = svm.SVC(C=C, gamma=gamma)
                svm_model.fit(new_x_train, y_train)
                score = svm_model.score(new_x_test, y_test)
                print('svm:C={} gamma:{} score={}'.format(C, gamma, score))
                if score > best['score']:
                    best['score'] = score
                    best['n_components'] = n_components
                    best['C'] = 4
                    best['gamma'] = gamma
    print('best paras:{}'.format(best))

train_for_model(digits)

    #
# kernel = ['rbf','poly','sigmoid']
# for k in kernel:
#     svc_model = svm.SVC(kernel=k)
#
#     svc_model.fit(X_train, y_train)
#
#     print(svc_model.score(X_test, y_test))