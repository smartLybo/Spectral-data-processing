import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,Birch

#   read .csv file  (data and lable)
original_data=np.loadtxt(r"./spectral_data/Siltstone 1.csv",skiprows=7,dtype=float,delimiter=',')
original_data=original_data[:,1:].T
all_lable=pd.read_csv(r"./spectral_data/Siltstone 1.csv",nrows=6,header=None)

#   tagging
light_source_zenith_angle_origin=all_lable.iloc[0,1:].values.T
light_source_zenith_angle_origin=pd.DataFrame(light_source_zenith_angle_origin,columns=['Light source zenith angle'])
light_source_zenith_angle_mapping = {'10':1, '20':2, '30':3, '40':4, '45':5, '50':6, '60':7, '70':8, '80':9}
light_source_zenith_angle_origin_new= light_source_zenith_angle_origin["Light source zenith angle"].map(light_source_zenith_angle_mapping)
light_source_zenith_angle_origin_new=light_source_zenith_angle_origin_new.to_numpy().T

Detect_zenith_angle_origin=all_lable.iloc[2,1:].values.T
Detect_zenith_angle_origin=pd.DataFrame(Detect_zenith_angle_origin,columns=['Detect zenith angle'])
Detect_zenith_angle_mapping = {'0':1, '5':2, '10':3, '15':4, '20':5, '25':6, '30':7, '35':8, '40':9
                               , '45':10, '50':11, '55':12, '60':13, '65':14, '70':15, '75':16, '80':17, '85':18}
Detect_zenith_angle_new=Detect_zenith_angle_origin["Detect zenith angle"].map(Detect_zenith_angle_mapping)
Detect_zenith_angle_new=Detect_zenith_angle_new.to_numpy().T

Detection_azimuth_origin=all_lable.iloc[4,1:].values.T
Detection_azimuth_origin=pd.DataFrame(Detection_azimuth_origin,columns=['Detection_azimuth'])
Detection_azimuth_mapping={'0':1, '10':2, '20':3, '30':4, '40':5}
Detection_azimuth_new=Detection_azimuth_origin["Detection_azimuth"].map(Detection_azimuth_mapping)
Detection_azimuth_new=Detection_azimuth_new.to_numpy().T

Granularity_origin=all_lable.iloc[5,1:].values.T
Granularity_origin=pd.DataFrame(Granularity_origin,columns=['Granularity'])
Granularity_mapping={'Block like with original surface':1, 'Block shape with polished surface':2,
                     '8mm':3, '4.75mm':4, '2.5mm':5, '1mm':6, '0.5mm':7, '0.21mm':8, '0.1mm':9, '0.074mm':10, '0.045mm':11}
Granularity_new=Granularity_origin["Granularity"].map(Granularity_mapping)
Granularity_new=Granularity_new.to_numpy().T

#   Reduce the number of features to 2D using PCA
pca=PCA(n_components=2)
data_pca=pca.fit_transform(original_data)

  # Visualization of feature distribution
plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=light_source_zenith_angle_origin_new)
plt.legend(handles=scatter.legend_elements(num=len(light_source_zenith_angle_mapping))[0],labels=['10°', '20°', '30°', '40°', '45°', '50°', '60°', '70°', '80°'],title="Angle")
plt.title('Different light source zenith angle')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different light source zenith angle.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=Detect_zenith_angle_new)
plt.legend(handles=scatter.legend_elements(num=len(Detect_zenith_angle_mapping))[0],labels=['0°', '5°', '10°', '15°', '20°', '25°', '30°', '35°', '40°',
                                                        '45°', '50°', '55°', '60°', '65°', '70°', '75°', '80°', '85°'],title="Angle")
plt.title('Different detect_zenith_angle')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different detect zenith angle.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=Detection_azimuth_new)
plt.legend(handles=scatter.legend_elements(num=len(Detection_azimuth_mapping))[0],labels=['0', '10', '20', '30', '40'],title="Angle")
plt.title('Different detection_azimuth')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different detection azimuth.jpg")
plt.show()

plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=Granularity_new)
plt.legend(handles=scatter.legend_elements(num=len(Granularity_mapping))[0],labels=['Block like with original surface', 'Block shape with polished surface',
                     '8mm', '4.75mm', '2.5mm', '1mm', '0.5mm', '0.21mm', '0.1mm', '0.074mm', '0.045mm'],title="Size")
plt.title('Different granularity')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different granularity.jpg")
plt.show()

#   Clustering original data with K-Means
birch = Birch(n_clusters=len(light_source_zenith_angle_mapping)).fit(data_pca)
plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=birch.labels_)
plt.title('Different light_source_zenith_angle after cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different light_source_zenith_angle after cluster.jpg")
plt.show()

birch = Birch(n_clusters=len(Detect_zenith_angle_mapping)).fit(original_data)
plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=birch.labels_)
plt.title('Different detect_zenith_angle after cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different detect zenith angle after cluster.jpg")
plt.show()

birch = Birch(n_clusters=len(Detection_azimuth_mapping)).fit(original_data)
plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=birch.labels_)
plt.title('Different detection azimuth after cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different detection azimuth after cluster.jpg")
plt.show()

birch = Birch(n_clusters=len(Granularity_mapping)).fit(original_data)
plt.figure(figsize=[8,7])
scatter=plt.scatter(data_pca[:,0], data_pca[:,1], c=birch.labels_)
plt.title('Different granularity after cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(r"./picture/Different granularity after cluster.jpg")
plt.show()

