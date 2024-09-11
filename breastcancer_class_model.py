import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim

''' INTERMEDIATE PROGRESS WITH 50 IMAGES '''
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
# from tensorflow.python.keras.models import ResNet50
# from tensorflow.python.keras.models import Model

# Load the metadata
meta_csv = './CBIS_DDSM/csv/meta.csv'
metadata_df = pd.read_csv(meta_csv)
print(metadata_df.head())


#TODO: 
# 1. read mass and calc train csv files and make X_train for mass and calc
# 2. make output csv folder that records "ID: ___ Type: ___" for each image (normal/benign/malignant) --> this is y_train
# 3. how to use resnet50?
# 4. data augmentation -> run the model without first and if it doesnt work then add data augmentation
# 5. eval and iterate
# 6. upload to github

# Load csv files
mass_train_csv = './CBIS_DDSM/csv/mass_case_description_train_set.csv'
calc_train_csv = './CBIS_DDSM/csv/calc_case_description_train_set.csv'
mass_test_csv = './CBIS_DDSM/csv/mass_case_description_test_set.csv'
calc_test_csv = './CBIS_DDSM/csv/calc_case_description_test_set.csv'

mass_train_df = pd.read_csv(mass_train_csv)
calc_train_df = pd.read_csv(calc_train_csv)

mass_patient_id = mass_train_df['patient_id']
calc_patient_id = calc_train_df['patient_id']

mass_test_df = pd.read_csv(mass_test_csv)
calc_test_df = pd.read_csv(calc_test_csv)

mass_test_patient_id = mass_test_df['patient_id']
calc_test_patient_id = calc_test_df['patient_id']

# Load dicom
dicom_csv = './CBIS_DDSM/csv/dicom_info.csv'
dicom_df = pd.read_csv(dicom_csv)

# Load image
# iterate patient id and read dicom file to find the corresponding image path, then load the image
# and store it in X_train
X_train_mass = []
mass_processed_ids = set()
for id in mass_patient_id.to_list():
    if id in mass_processed_ids:
        continue  # Skip if id has already been processed
    else:
        filtered_df = dicom_df[dicom_df['PatientID'].str.contains(id)]
        img_path = filtered_df['image_path']
    
    for img in img_path:
        corrected_img = img.replace('CBIS-DDSM', 'CBIS_DDSM')
        corrected_path = os.path.join('./', corrected_img)
        X_train_mass.append(corrected_path)
    
    mass_processed_ids.add(id)  # Mark this id as processed

y_train_mass = mass_train_df['pathology']
y_train_mass = np.array(y_train_mass)

X_test_mass = []
mass_test_processed_ids = set()
for id in mass_test_patient_id:
    if id in mass_test_processed_ids:
        continue  # Skip if id has already been processed
    filtered_df = dicom_df[dicom_df['PatientID'].str.contains(id)]
    img_path = filtered_df['image_path']
    
    for img in img_path:
        corrected_img = img.replace('CBIS-DDSM', 'CBIS_DDSM')
        corrected_path = os.path.join('./', corrected_img)
        X_train_mass.append(corrected_path)
    
    mass_test_processed_ids.add(id)  # Mark this id as processed

y_test_mass = mass_test_df['pathology']
y_test_mass = np.array(y_test_mass)

X_train_calc = []
calc_processed_ids = set()
for id in calc_patient_id:
    if id in calc_processed_ids:
        continue  # Skip if id has already been processed
    filtered_df = dicom_df[dicom_df['PatientID'].str.contains(id)]
    img_path = filtered_df['image_path']
    
    for img in img_path:
        corrected_img = img.replace('CBIS-DDSM', 'CBIS_DDSM')
        corrected_path = os.path.join('./', corrected_img)
        X_train_mass.append(corrected_path)
    
    calc_processed_ids.add(id)  # Mark this id as processed

y_train_calc = calc_train_df['pathology']
y_train_calc = np.array(y_train_calc)
# print(X_train_calc)

X_test_calc = []
calc_test_processed_ids = set()
for id in calc_test_patient_id:
    if id in calc_test_processed_ids:
        continue  # Skip if id has already been processed
    filtered_df = dicom_df[dicom_df['PatientID'].str.contains(id)]
    img_path = filtered_df['image_path']
    
    for img in img_path:
        corrected_img = img.replace('CBIS-DDSM', 'CBIS_DDSM')
        corrected_path = os.path.join('./', corrected_img)
        X_train_mass.append(corrected_path)
    
    calc_test_processed_ids.add(id)  # Mark this id as processed

y_test_calc = calc_test_df['pathology']
y_test_calc = np.array(y_test_calc)

X_train_mass_txt = './X_train_mass.txt'
X_test_mass_txt = './X_test_mass.txt'
y_train_mass_txt = './y_train_mass.txt'
y_test_mass_txt = './y_test_mass.txt'

X_train_calc_txt = './X_train_calc.txt'
X_test_calc_txt = './X_test_calc.txt'
y_train_calc_txt = './y_train_calc.txt'
y_test_calc_txt = './y_test_calc.txt'

""" save everything as txt files """
if not os.path.exists(X_train_mass_txt):
    np.savetxt(X_train_mass_txt, X_train_mass, fmt='%s')
if not os.path.exists(X_test_mass_txt):
    np.savetxt(X_test_mass_txt, X_test_mass, fmt='%s')
if not os.path.exists(y_train_mass_txt):
    np.savetxt(y_train_mass_txt, y_train_mass, fmt='%s')
if not os.path.exists(y_test_mass_txt):
    np.savetxt(y_test_mass_txt, y_test_mass, fmt='%s')

if not os.path.exists(X_train_calc_txt):
    np.savetxt(X_train_calc_txt, X_train_calc, fmt='%s')
if not os.path.exists(X_test_calc_txt):
    np.savetxt(X_test_calc_txt, X_test_calc, fmt='%s')
if not os.path.exists(y_train_calc_txt):
    np.savetxt(y_train_calc_txt, y_train_calc, fmt='%s')
if not os.path.exists(y_test_calc_txt):
    np.savetxt(y_test_calc_txt, y_test_calc, fmt='%s')

# Load images
# X = image path
# y = label
# def load_images(img_path_list):
#     img_vec = []
#     # Iterate through each line in the file
#     for i, img_path in enumerate(img_path_list):
#     # Process the line
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (224, 224))
#         img = img / 255.0
#         img = np.expand_dims(img, axis=0)
#         img_vec.append(img)
#     return np.array(img_vec)
# this should process images so that it gets fed into the model

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

# currently only loading 50 images for testing
X_train_mass_img = load_images(X_train_mass[:50])
y_train_mass_label = y_train_mass[:50]

# print(X_train_mass_img.shape)
# X_train_mass_img = np.expand_dims(X_train_mass_img, axis=-1)
# y_train_mass_label = to_categorical(y_train_mass_label, num_classes=3) # classes: benign, benign without callback, malignant

X_train_calc_img = load_images(X_train_calc[:50])
y_train_calc_label = y_train_calc[:50]

# X_train_calc_img = np.expand_dims(X_train_calc_img, axis=-1)
# y_train_calc_label = to_categorical(y_train_calc_label, num_classes=3)

X_test_mass_img = load_images(X_test_mass[:50])
y_test_mass_label = y_test_mass[:50]

# X_test_mass_img = np.expand_dims(X_test_mass_img, axis=-1)
# y_test_mass_label = to_categorical(y_test_mass_label, num_classes=3)

X_test_calc_img = load_images(X_test_calc)
y_test_calc_label = y_test_calc

# X_test_calc_img = np.expand_dims(X_test_calc_img, axis=-1)
# y_test_calc_label = to_categorical(y_test_calc_label, num_classes=3)

# Split the data
# data already split according to mass/calc

def create_dataloader(X_img, y_label, batch_size = 4, shuffle = True):

    label_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT': 2}

# Assuming y_label is a list or array of string labels
    y_label_int = [label_mapping[label] for label in y_label]
    y_label_tensor = torch.tensor(y_label_int).long()
    dataset = TensorDataset(X_img, y_label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# dataloaders
# train val split happens here
mass_train_loader = create_dataloader(X_train_mass_img, y_train_mass_label)
mass_test_loader = create_dataloader(X_test_mass_img, y_test_mass_label)
calc_train_loader = create_dataloader(X_train_calc_img, y_train_calc_label)
calc_test_loader = create_dataloader(X_test_calc_img, y_test_calc_label)

from torch.utils.data import ConcatDataset

combined_train_dataset = ConcatDataset([mass_train_loader.dataset, calc_train_loader.dataset])

train_len = int(0.9*len(combined_train_dataset))
val_len = len(combined_train_dataset) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(combined_train_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

combined_test_dataset = ConcatDataset([mass_test_loader.dataset, calc_test_loader.dataset])
combined_test_loader = DataLoader(combined_test_dataset, batch_size=4, shuffle=True)

dataloaders = {'train': train_loader, 'val': val_loader}

# Model
model_ft = models.resnet50(pretrained=True)
model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 3) # 3 classes: benign, benign without callback, malignant

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# train
def train(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# dataloaders
# combine mass and calc train and test loaders

trained_model = train(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=25)

# evaluate
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

evaluate(trained_model, combined_test_loader)

# save model
torch.save(trained_model.state_dict(), 'breast_cancer_model.pth')

def load_model(model, path='breast_cancer_model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_image_class(model, img_path_list, device):
    """
    Predicts the class name for a given image.

    Args:
        model: Trained model for prediction.
        img_path: Path to the image file.
        device: Device to perform inference on ('cpu' or 'cuda').

    Returns:
        str: Predicted class name.
    """
    # define prediction class list
    predict_class_list = []

    # Load and preprocess the image
    img_vec = load_images(img_path_list)
    img_vec_tensor = torch.tensor(img_vec).float().to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        output = model(img_vec_tensor)
        _, pred = torch.max(output, 1)

    # Map numerical prediction to class label
    class_labels = ['BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT']
    for entry in pred:
        predicted_class = class_labels[pred[entry].item()]
        predict_class_list.append(predicted_class)

    return predict_class_list

# Load the trained model
# predict

model_trained = load_model(trained_model, 'breast_cancer_model.pth')
X_test_mass_predict = predict_image_class(model_trained, X_test_mass[:50], device)
X_test_calc_predict = predict_image_class(model_trained, X_test_calc[:50], device)

# compare prediction with actual label
# calculate accuracy
# confusion matrix (integrate later)

accuracy = evaluate(model_trained, combined_test_loader)
print("Model accuracy: ", accuracy)

with open('X_test_mass_predict.txt', 'w') as f:
    for item in X_test_mass_predict:
        f.write("%s\n" % item)
with open('X_test_calc_predict.txt', 'w') as f:
    for item in X_test_calc_predict:
        f.write("%s\n" % item)