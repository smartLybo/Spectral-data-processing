import numpy as np

original_data_1=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/0.3 coking coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_1=original_data_1[:,1:]
original_data_2=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Anthracite coal grade 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_2=original_data_2[:,1:]
original_data_3=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Anthracite coal grade 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_3=original_data_3[:,1:]
original_data_4=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Arenaceous shale 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_4=original_data_4[:,1:]
original_data_5=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Arenaceous shale 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_5=original_data_5[:,1:]
original_data_6=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Argillaceous limestone 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_6=original_data_6[:,1:]
original_data_7=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Argillaceous limestone 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_7=original_data_7[:,1:]
original_data_8=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Black shale 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_8=original_data_8[:,1:]
original_data_9=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Black shale 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_9=original_data_9[:,1:]
original_data_10=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Carbonaceous shale.csv",skiprows=7,dtype=float,delimiter=',')
original_data_10=original_data_10[:,1:]
original_data_11=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Clay.csv",skiprows=7,dtype=float,delimiter=',')
original_data_11=original_data_11[:,1:]
original_data_12=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Coking coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_12=original_data_12[:,1:]
original_data_13=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Fat coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_13=original_data_13[:,1:]
original_data_14=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Fine-grained sandstone.csv",skiprows=7,dtype=float,delimiter=',')
original_data_14=original_data_14[:,1:]
original_data_15=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Gas coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_15=original_data_15[:,1:]
original_data_16=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Gas-fat coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_16=original_data_16[:,1:]
original_data_17=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Lean coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_17=original_data_17[:,1:]
original_data_18=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Lean-thin coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_18=original_data_18[:,1:]
original_data_19=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Lignite coal grade 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_19=original_data_19[:,1:]
original_data_20=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Lignite coal grade 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_20=original_data_20[:,1:]
original_data_21=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Medium-grained sandstone.csv",skiprows=7,dtype=float,delimiter=',')
original_data_21=original_data_21[:,1:]
original_data_22=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Siltstone 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data_22=original_data_22[:,1:]
original_data_23=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Siltstone 2.csv",skiprows=7,dtype=float,delimiter=',')
original_data_23=original_data_23[:,1:]
original_data_24=np.loadtxt(r"E:/论文/小论文/Scientific Data——光谱数据库/Spectroscopy dataset/Spectrum/Thin coal.csv",skiprows=7,dtype=float,delimiter=',')
original_data_24=original_data_24[:,1:]


original_data=np.concatenate((original_data_1,original_data_2,original_data_3,original_data_4,
                              original_data_5,original_data_6,original_data_7,original_data_8,
                              original_data_9,original_data_10,original_data_11,original_data_12,
                              original_data_13,original_data_14,original_data_15,original_data_16,
                              original_data_17,original_data_18,original_data_19,original_data_20,
                              original_data_21,original_data_22,original_data_23,original_data_24
                              ),axis=1)
print(original_data.shape)