import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#   Read data
original_data_1=np.loadtxt(r"./spectral_data/Siltstone 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_1=original_data_1[:,1:]
original_data_2=np.loadtxt(r"./spectral_data/Arenaceous shale 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_2=original_data_2[:,1:]
original_data_3=np.loadtxt(r"./spectral_data/Gas coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_3=original_data_3[:,1:]
original_data_4=np.loadtxt(r"./spectral_data/Lean coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_4=original_data_4[:,1:]
original_data=np.concatenate((original_data_1,original_data_2,original_data_3,original_data_4),axis=1)

#   make label
lable_zero=np.expand_dims(np.full(shape=original_data_1.shape[1],fill_value=0),axis=1)
lable_one=np.expand_dims(np.full(shape=original_data_2.shape[1],fill_value=0),axis=1)
lable_two=np.expand_dims(np.full(shape=original_data_3.shape[1],fill_value=0),axis=1)
lable_three=np.expand_dims(np.full(shape=original_data_4.shape[1],fill_value=0),axis=1)
lable_classification_two=np.concatenate((lable_zero+1,lable_one+1,lable_two,lable_three),axis=0)
lable_classification_mul=np.concatenate((lable_zero,lable_one+1,lable_two+2,lable_three+3),axis=0)
lable_classification_two=lable_classification_two.flatten()
lable_classification_mul=lable_classification_mul.flatten()


#   data standardization
scaler = StandardScaler()
stand_data=scaler.fit_transform(original_data.T)


#   Dataset partitioning
#two
X_train_two,X_test_two_origin,Y_train_two,Y_test_two_origin=train_test_split(stand_data,lable_classification_two,test_size=0.8,random_state=10)
X_test_two,X_verify_two,Y_test_two,Y_verify_two=train_test_split(X_test_two_origin,Y_test_two_origin,test_size=0.5,random_state=0)
#mul
X_train_mul,X_test_mul_origin,Y_train_mul,Y_test_mul_origin=train_test_split(stand_data,lable_classification_mul,test_size=0.8,random_state=10)
X_test_mul,X_verify_mul,Y_test_mul,Y_verify_mul=train_test_split(X_test_mul_origin,Y_test_mul_origin,test_size=0.5,random_state=0)

#   Classification recognition train
svc_two=SVC(kernel="rbf")
svc_two.fit(X_train_two,Y_train_two)
Y_pre_two=svc_two.predict(X_test_two)
print("The score for the two category F1 in test is %.3f" %f1_score(Y_test_two,Y_pre_two))
print("The score for the two category ACC in test is %.3f" %accuracy_score(Y_test_two,Y_pre_two))
print("")

forest =RandomForestClassifier(n_estimators=50, random_state=2)
forest.fit(X_train_mul,Y_train_mul)
Y_pre_mul=forest.predict(X_test_mul)
print("The score for the mul category F1 in test is %.3f" %f1_score(Y_test_mul,Y_pre_mul,average='weighted'))
print("The score for the mul category ACCin test  is %.3f" %accuracy_score(Y_test_mul,Y_pre_mul))
print("")

# Classification recognition verify
Y_pre_two=svc_two.predict(X_verify_two)
print("The score for the two category F1 in verify is %.3f" %f1_score(Y_verify_two,Y_pre_two))
print("The score for the two category ACC in verify is %.3f" %accuracy_score(Y_verify_two,Y_pre_two))
print("")
Y_pre_mul=forest.predict(X_verify_mul)
print("The score for the mul category F1 in verify is %.3f" %f1_score(Y_verify_mul,Y_pre_mul,average='weighted'))
print("The score for the mul category ACC in verify is %.3f" %accuracy_score(Y_verify_mul,Y_pre_mul))
print("")

#   Draw a scatter plot
pca_two=PCA(n_components=2)
data_pca_two=pca_two.fit_transform(X_verify_two)

pca_mul=PCA(n_components=4)
data_pca_mul=pca_mul.fit_transform(X_verify_mul)

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca_two[:,0], data_pca_two[:,1], c=Y_verify_two)
plt.legend(handles=scatter.legend_elements(num=2)[0],labels=['coal', 'rock',],title="Category")
plt.title('Real binary classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Real binary classification.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca_two[:,0], data_pca_two[:,1], c=Y_pre_two)
plt.legend(handles=scatter.legend_elements(num=2)[0],labels=['coal', 'rock',],title="Category")
plt.title('Predict binary classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Predict binary classification.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca_mul[:,0], data_pca_mul[:,1], c=Y_verify_mul)
plt.legend(handles=scatter.legend_elements(num=4)[0],labels=['Siltstone 1', 'Arenaceous shale 1',"Gas coal","Lean coal"],title="Category")
plt.title('Real multi classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Real multi classification.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca_mul[:,0], data_pca_mul[:,1], c=Y_pre_mul)
plt.legend(handles=scatter.legend_elements(num=4)[0],labels=['Siltstone 1', 'Arenaceous shale 1',"Gas coal","Lean coal"],title="Category")
plt.title('Predict multi classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Predict multi classification.jpg")
plt.show()

