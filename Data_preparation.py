import numpy as np
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.model_selection import train_test_split

#   Read data
def Read_data(type):
    # get reflection spectrum and make lable
    Sample_List=["0.3 coking coal","Anthracite coal grade 1","Anthracite coal grade 2","Arenaceous shale 1"
                 ,"Arenaceous shale 2","Argillaceous limestone 1","Argillaceous limestone 2","Black shale 1"
                 ,"Black shale 2","Carbonaceous shale","Clay","Coking coal"
                 ,"Fat coal","Fine-grained sandstone","Gas coal","Gas-fat coal"
                 ,"Lean coal","Lean-thin coal","Lignite coal grade 1","Lignite coal grade 2"
                 ,"Medium-grained sandstone","Siltstone 1","Siltstone 2","Thin coal"]
    coal_name=["Anthracite coal grade 1","Anthracite coal grade 2","Lean coal","Lean-thin coal",
               "Thin coal","Coking coal","Fat coal","0.3 coking coal",
               "Gas-fat coal","Gas coal","Lignite coal grade 1","Lignite coal grade 2"]
    coal_list=[0,1,2,11,12,14,15,16,17,18,19,23]
    rock_list=[3,4,5,6,7,8,9,10,13,20,21,22]
    all_original_data_list=[]    # Original reflection spectrum
    all_bin_label_list = []  # Binary labels
    all_mul_lable_list = []  # Multi labels
    all_qua_lable_list = []  # Quantitative labels
    myl=0
    for i in range(len(Sample_List)):
        original_data = np.loadtxt(r"./Spectroscopy dataset/Spectrum/"+Sample_List[i]+".csv", skiprows=7, dtype=float, delimiter=',')[:, 1:]
        all_original_data_list.append(original_data)
        if i in coal_list:
            bin_array=np.zeros(original_data.shape[1])
            all_bin_label_list.append(bin_array)

            mul_array=np.zeros(original_data.shape[1])+myl
            myl=myl+1
            all_mul_lable_list.append(mul_array)

            original_ICA_data = np.loadtxt(r"./Spectroscopy dataset/Analysis/ICA.csv", skiprows=1,usecols=range(1, 5),dtype=float, delimiter=',')
            index_position = coal_name.index(Sample_List[i])
            ash_value=original_ICA_data[index_position,1]
            qua_array=np.zeros(original_data.shape[1])+ash_value
            all_qua_lable_list.append(qua_array)
        elif i in rock_list:
            bin_array = np.zeros(original_data.shape[1]) + 1
            all_bin_label_list.append(bin_array)

    all_original_data=np.concatenate(all_original_data_list,axis=1).T
    coal_data=np.concatenate([all_original_data_list[j] for j in coal_list] ,axis=1).T

    all_bin_label = np.expand_dims(np.concatenate(all_bin_label_list,axis=0),axis=1)
    all_mul_lable = np.expand_dims(np.concatenate(all_mul_lable_list,axis=0),axis=1)
    all_qua_lable = np.expand_dims(np.concatenate(all_qua_lable_list,axis=0),axis=1)

    #分类时使用
    encoder = OneHotEncoder(sparse=False)
    all_bin_label = encoder.fit_transform(all_bin_label)
    all_mul_lable = encoder.fit_transform(all_mul_lable)

    if type=="bin":
        return [all_original_data,all_bin_label]
    elif type=="mul":
        return [coal_data,all_mul_lable]
    else:
        return [coal_data,all_qua_lable]

def standardization(data):      # data standardization
    scaler = StandardScaler()
    stand_data=scaler.fit_transform(data)
    return stand_data

def data_partitioning(data_lable):    #Dataset partiioning
    X_train,X_test,Y_train,Y_test=train_test_split(data_lable[0],data_lable[1],test_size=0.3,shuffle=True)
    X_test,X_verify,Y_test,Y_verify=train_test_split(X_test,Y_test,test_size=0.4,shuffle=True)
    return [[X_train,Y_train],[X_test,Y_test],[X_verify,Y_verify]]

