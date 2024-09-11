import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load the metadata
meta_csv = './CBIS_DDSM/csv/meta.csv'
metadata_df = pd.read_csv(meta_csv)
# print(metadata_df.head())

# Load csv files
mass_train_csv = './CBIS_DDSM/csv/mass_case_description_train_set.csv'
calc_train_csv = './CBIS_DDSM/csv/calc_case_description_train_set.csv'
mass_test_csv = './CBIS_DDSM/csv/mass_case_description_test_set.csv'
calc_test_csv = './CBIS_DDSM/csv/calc_case_description_test_set.csv'

mass_train_df = pd.read_csv(mass_train_csv)
calc_train_df = pd.read_csv(calc_train_csv)

mass_test_df = pd.read_csv(mass_test_csv)
calc_test_df = pd.read_csv(calc_test_csv)

mass_patient_id = mass_train_df['cropped image file path'].str.split('/').str[1].tolist()
calc_patient_id = calc_train_df['cropped image file path'].str.split('/').str[1].tolist()
mass_test_patient_id = mass_test_df['cropped image file path'].str.split('/').str[1].tolist()
calc_test_patient_id = calc_test_df['cropped image file path'].str.split('/').str[1].tolist()
pat_uid = mass_patient_id + calc_patient_id + mass_test_patient_id + calc_test_patient_id # in order of mass_train, calc_train, mass_test, calc_test

mass_patient_pathology = mass_train_df['pathology'].tolist()
calc_patient_pathology = calc_train_df['pathology'].tolist()
mass_test_patient_pathology = mass_test_df['pathology'].tolist()
calc_test_patient_pathology = calc_test_df['pathology'].tolist()

""" LABEL CHART """
# 0 = benign without callback, 1 = benign, 2 = malignant
""" LABEL CHART """

def convert_pathology_to_label(pathology):
    if pathology == 'BENIGN':
        return 0
    elif pathology == 'BENIGN_WITHOUT_CALLBACK':
        return 1
    elif pathology == 'MALIGNANT':
        return 2
    else:
        return -1
    
def label_pathology(pathology_list):
    return [convert_pathology_to_label(x) for x in pathology_list]

mass_train_labels = label_pathology(mass_patient_pathology)
calc_train_labels = label_pathology(calc_patient_pathology)
mass_test_labels = label_pathology(mass_test_patient_pathology)
calc_test_labels = label_pathology(calc_test_patient_pathology)

# load y labels first
# y labels are pathology labels
# order: mass train, calc train, mass test, calc test
y = []
y.extend(mass_train_labels)
y.extend(calc_train_labels)
y.extend(mass_test_labels)
y.extend(calc_test_labels)

print("here")

dicom_csv = './CBIS-DDSM/csv/dicom_info.csv'
dicom_df = pd.read_csv(dicom_csv)
pat_uid_all = dicom_df['StudyInstanceUID'].tolist() # use UID instead of PatientID because PatientID is not unique
labeled_pat_uid = set(pat_uid) & set(pat_uid_all)
# matching_rows = df2[df2['CommonColumn'].isin(overlap)]
# other_column_values = matching_rows['OtherColumn'].tolist()
matching_rows = dicom_df[dicom_df['StudyInstanceUID'].isin(labeled_pat_uid)]
og_image_path_list = matching_rows['image_path'].tolist()
image_path_list = []
for entry in og_image_path_list:
    corrected_img = og_image_path_list[entry].replace('CBIS-DDSM', 'CBIS_DDSM')
    corrected_path = os.path.join('./', corrected_img)
    image_path_list.append(corrected_path)

# load images but return in tensor format
def load_images(img_path_list, device='cuda'):
    img_tensors = []
    it_num = 0
    
    # Iterate through each image path in the list
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        img = img / 255.0  # Normalize the image to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add channel dimension (1, H, W)
        
        img_tensor = torch.tensor(img, dtype=torch.float32)  # Convert numpy array to tensor
        img_tensors.append(img_tensor)
        print(it_num)
        it_num += 1

    # Stack all tensors into a single tensor with shape [N, 1, 224, 224]
    img_tensors = torch.stack(img_tensors)
    
    # Move the tensor to the specified device (GPU or CPU)
    # img_tensors = img_tensors.to(device)
    
    return img_tensors

# Load the images
X = []


# Load labels
# 0 = benign without callback, 1 = benign, 2 = malignant
y = []
mass_train = 'Mass-Training'
calc_train = 'Calc-Training'
mass_test = 'Mass-Test'
calc_test = 'Calc-Test'
for index in range(len(image_path_list)):
    image_path = image_path_list[index]
    patient_id_str = dicom_df['PatientID'][index]

