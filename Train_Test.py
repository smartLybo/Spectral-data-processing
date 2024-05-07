from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
from hpelm import ELM
import Binary_classification_model as binc
import Mul_classification_model as mulc
import Quantitative_model as quar
import original_BLS
import Data_preparation as DP
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch

# epochs=150

## Bin
# for i in range(10):
#     bi_data = DP.Read_data("bin")
#     std_data = DP.standardization(bi_data[0])
#     out_data = DP.data_partitioning([std_data,bi_data[1]])
#     data_merge = torch.from_numpy(out_data[0][0]).type(torch.float32)   # Retrieve training data
#     lable_merge = torch.from_numpy(out_data[0][1]).type(torch.float32)
#     dataset = TensorDataset(data_merge, lable_merge)
#     train_loader = DataLoader(dataset, batch_size=500, shuffle=True)
#
#     CNN=binc.CNN()
#     BiLSTM=binc.BiLSTM()
#     BLS_bin=original_BLS.broadnet_enhmap()
#
#     criterion_list=[]
#     optimizer_list=[]
#     criterion_list.append(nn.CrossEntropyLoss())
#     criterion_list.append(nn.CrossEntropyLoss())
#     optimizer_list.append(optim.Adam(CNN.parameters(), lr=0.0001))
#     optimizer_list.append(optim.Adam(BiLSTM.parameters(), lr=0.0001))
#
#     steps_per_epoch=len(train_loader)
#
#     for epoch in range(epochs):
#         steps = 0
#         for data, lable in train_loader:
#             optimizer_list[0].zero_grad()
#             optimizer_list[1].zero_grad()
#             output_1 = CNN(data.unsqueeze(1))
#             output_2 = BiLSTM(data)
#             loss_train_1 = criterion_list[0](output_1, lable)
#             loss_train_2 = criterion_list[1](output_2, lable)
#             loss_1=loss_train_1.detach()
#             loss_2 = loss_train_2.detach()
#             loss_train_1.backward()
#             loss_train_2.backward()
#             optimizer_list[0].step()
#             optimizer_list[1].step()
#             steps = steps + 1
#             # print('Epoch:', '%03d' % (epoch + 1), ', Step:', '%03d' % (steps), '/', '%03d' % (steps_per_epoch),
#             #       ', loss_1 =', '{:.6f}'.format(loss_1),  ', loss_2 =',
#             #       '{:.6f}'.format(loss_2))
#     BLS_bin.fit(out_data[0][0],out_data[0][1])
#
#     #test data
#     test_input_data_CNN=torch.from_numpy(out_data[1][0]).type(torch.float32).unsqueeze(1)
#     y_pre_CNN=CNN(test_input_data_CNN)
#     acc_CNN=binc.show_accuracy(y_pre_CNN,out_data[1][1])
#     print('the accuracy of cnn in test is {:.6f}'.format(acc_CNN))
#     test_input_data_BiLSTM=torch.from_numpy(out_data[1][0]).type(torch.float32)
#     y_pre_BiLSTM=BiLSTM(test_input_data_BiLSTM)
#     acc_BiLSTM = binc.show_accuracy(y_pre_BiLSTM,out_data[1][1])
#     print('the accuracy of BiLSTM in test  is {:.6f}'.format(acc_BiLSTM))
#     y_pre_BLS_bin=BLS_bin.predict(out_data[1][0])
#     acc_BLS = binc.show_accuracy(y_pre_BLS_bin,out_data[1][1])
#     print('the accuracy of BLS in test  is {:.6f}'.format(acc_BLS))
#     print(' ')
#
#     #Validation data
#     test_input_data_CNN=torch.from_numpy(out_data[2][0]).type(torch.float32).unsqueeze(1)
#     y_pre_CNN=CNN(test_input_data_CNN)
#     acc_CNN=binc.show_accuracy(y_pre_CNN,out_data[2][1])
#     print('the accuracy of cnn in validation is {:.6f}'.format(acc_CNN))
#     test_input_data_BiLSTM=torch.from_numpy(out_data[2][0]).type(torch.float32)
#     y_pre_BiLSTM=BiLSTM(test_input_data_BiLSTM)
#     acc_BiLSTM = binc.show_accuracy(y_pre_BiLSTM,out_data[2][1])
#     print('the accuracy of BiLSTM in validation is {:.6f}'.format(acc_BiLSTM))
#     y_pre_BLS_bin=BLS_bin.predict(out_data[2][0])
#     acc_BLS = binc.show_accuracy(y_pre_BLS_bin,out_data[2][1])
#     print('the accuracy of BLS in validation is {:.6f}'.format(acc_BLS))
#     print('----------------------')

## Mul
# for i in range(10):
#     mul_data = DP.Read_data("mul")
#     std_mul_data = DP.standardization(mul_data[0])
#     out_mul_data = DP.data_partitioning([std_mul_data,mul_data[1]])
#
#     svm = SVC(C=400)
#     lda = LinearDiscriminantAnalysis()
#     rf = RandomForestClassifier(n_estimators=300)
#     pca = PCA(n_components=10)
#     pca.fit(out_mul_data[0][0])
#     pca_components = pca.components_
#
#     pca_mul_data = np.dot(out_mul_data[0][0], pca_components.T)
#     svm.fit(pca_mul_data,out_mul_data[0][1].ravel())
#     lda.fit(pca_mul_data,out_mul_data[0][1].ravel())
#     rf.fit(pca_mul_data,out_mul_data[0][1].ravel())
#
#     #test data
#     pca_mul_test_data = np.dot(out_mul_data[1][0], pca_components.T)
#     y_pre_test_svm=svm.predict(pca_mul_test_data)
#     acc_svm=mulc.show_accuracy(y_pre_test_svm,out_mul_data[1][1])
#     print('the accuracy of svm in test is {:.6f}'.format(acc_svm))
#     y_pre_test_lda=lda.predict(pca_mul_test_data)
#     acc_lda=mulc.show_accuracy(y_pre_test_lda,out_mul_data[1][1])
#     print('the accuracy of lda in test is {:.6f}'.format(acc_lda))
#     y_pre_test_rf=rf.predict(pca_mul_test_data)
#     acc_rf=mulc.show_accuracy(y_pre_test_rf,out_mul_data[1][1])
#     print('the accuracy of rf in test is {:.6f}'.format(acc_rf))
#
#     #Validation data
#     pca_mul_validation_data = np.dot(out_mul_data[2][0], pca_components.T)
#     y_pre_test_svm=svm.predict(pca_mul_validation_data)
#     acc_svm=mulc.show_accuracy(y_pre_test_svm,out_mul_data[2][1])
#     print('the accuracy of svm in validation is {:.6f}'.format(acc_svm))
#     y_pre_test_lda=lda.predict(pca_mul_validation_data)
#     acc_lda=mulc.show_accuracy(y_pre_test_lda,out_mul_data[2][1])
#     print('the accuracy of lda in validation is {:.6f}'.format(acc_lda))
#     y_pre_test_rf=rf.predict(pca_mul_validation_data)
#     acc_rf=mulc.show_accuracy(y_pre_test_rf,out_mul_data[2][1])
#     print('the accuracy of rf in validation is {:.6f}'.format(acc_rf))
#     print('--------------------')

# for i in range(10):
#     mul_data = DP.Read_data("mul")
#     std_mul_data = DP.standardization(mul_data[0])
#     out_mul_data = DP.data_partitioning([std_mul_data,mul_data[1]])
#
#     elm = ELM(inputs=1500, outputs=12)
#     elm.add_neurons(1000, 'tanh')
#     elm.add_neurons(400, 'rbf_l2')
#     elm.add_neurons(50, 'lin')
#     BLS_mul=original_BLS.broadnet_enhmap()
#
#     elm.train(out_mul_data[0][0],out_mul_data[0][1])
#     BLS_mul.fit(out_mul_data[0][0],out_mul_data[0][1])
#
#     y_pre_test_elm=elm.predict(out_mul_data[1][0])
#     acc_elm=mulc.show_accuracy_hot(y_pre_test_elm,out_mul_data[1][1])
#     print('the accuracy of elm in test is {:.6f}'.format(acc_elm))
#     y_pre_test_BLS_mul=BLS_mul.predict(out_mul_data[1][0])
#     acc_BLS=mulc.show_accuracy_hot(y_pre_test_BLS_mul,out_mul_data[1][1])
#     print('the accuracy of BLS in test is {:.6f}'.format(acc_BLS))
#
#     y_pre_test_elm=elm.predict(out_mul_data[2][0])
#     acc_elm=mulc.show_accuracy_hot(y_pre_test_elm,out_mul_data[2][1])
#     print('the accuracy of elm in validation is {:.6f}'.format(acc_elm))
#     y_pre_test_BLS_mul=BLS_mul.predict(out_mul_data[2][0])
#     acc_BLS=mulc.show_accuracy_hot(y_pre_test_BLS_mul,out_mul_data[2][1])
#     print('the accuracy of BLS in validation is {:.6f}'.format(acc_BLS))
#     print('-------------------------------')



## Quantitative
for i in range(10):
    qua_data = DP.Read_data("qua")
    std_qua_data = DP.standardization(qua_data[0])
    out_qua_data = DP.data_partitioning([std_qua_data,qua_data[1]])

    xgb = XGBRegressor()
    pls = PLSRegression(n_components=10)
    pca = PCA(n_components=10)
    pca.fit(out_qua_data[0][0])
    pca_components = pca.components_

    pca_mul_data = np.dot(out_qua_data[0][0], pca_components.T)
    xgb.fit(pca_mul_data,out_qua_data[0][1])
    pls.fit(pca_mul_data,out_qua_data[0][1])

    #test data
    pca_mul_test_data = np.dot(out_qua_data[1][0], pca_components.T)
    y_pre_test_xgb=xgb.predict(pca_mul_test_data)
    acc_xgb=quar.get_MAE(y_pre_test_xgb,out_qua_data[1][1])
    print('the MAE of xgb in test is {:.6f}'.format(acc_xgb))
    y_pre_test_pls=pls.predict(pca_mul_test_data)
    acc_pls=quar.get_MAE(y_pre_test_pls,out_qua_data[1][1])
    print('the MAE of pls in test is {:.6f}'.format(acc_pls))

    #Validation data
    pca_mul_validation_data = np.dot(out_qua_data[2][0], pca_components.T)
    y_pre_test_xgb=xgb.predict(pca_mul_validation_data)
    acc_xgb=quar.get_MAE(y_pre_test_xgb,out_qua_data[2][1])
    print('the MAE of xgb in validation is {:.6f}'.format(acc_xgb))
    y_pre_test_pls=pls.predict(pca_mul_validation_data)
    acc_pls=quar.get_MAE(y_pre_test_pls,out_qua_data[2][1])
    print('the MAE of pls in validation is {:.6f}'.format(acc_pls))
    print('-------------------------------------')

