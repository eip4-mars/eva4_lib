import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import traceback
import warnings
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

def create_dataset_csv_split(path, test_ratio = 0.2, out_path = '/content/'):
    warnings.filterwarnings("error")
    if not path.endswith(os.path.sep):out_path += os.path.sep
    if not out_path.endswith(os.path.sep):out_path += os.path.sep
    classes = os.listdir(path)
    train = []
    test = []
    labels = pd.DataFrame(classes,index=range(len(classes)), columns=['Label'])
    labels.reset_index(inplace=True)

    #print(labels)
    for i in range(len(classes)):
        child_path = path+classes[i]+'/'
        imgs = os.listdir(child_path)
        #print(len(tr),classes[i])
        trn,tst = train_test_split(imgs, test_size=test_ratio, random_state=42)
        
        train += [[child_path+x, i] for x in trn]
        test += [[child_path+x, i] for x in tst]

    # train1 = train.copy()
    # test1 = test.copy()

    for impath in tqdm(train):
        try:
            im = Image.open(impath[0])
            im.close()
        except:
            train.remove(impath)
            pass

    for impath in tqdm(test):
        try:
            im = Image.open(impath[0])
            im.close()
        except:
            test.remove(impath)
            pass

    train_df = pd.DataFrame(train,columns=['File_name', 'label_index'])
    test_df = pd.DataFrame(test,columns=['File_name', 'label_index'])

    print("Train shape : ", train_df.shape, ", Test shape : ", test_df.shape, ", Labels shape : ", labels.shape)
    #train_df_1 = pd.concat([train_df[["File_name"]], pd.get_dummies(train_df.label_index, prefix="label")], axis = 1)
    #test_df_1 = pd.concat([test_df[["File_name"]], pd.get_dummies(test_df.label_index, prefix="label")], axis = 1)

    train_df.to_csv(out_path+'train.csv',index=False)
    test_df.to_csv(out_path+'test.csv',index=False)
    labels.to_csv(out_path+'labels.csv',index=False)
    print('Files (train.csv, test.csv and labels.csv) generated in : ',out_path)
    warnings.filterwarnings("ignore")


class custDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, csv_file, transform=None):
		self.frame = pd.read_csv(csv_file, header=0)
		self.transform = transform

	def __getitem__(self, idx):
		image_name = self.frame.iloc[idx, 0]
		label = self.frame.iloc[idx, 1]
		
		image = Image.open(image_name).convert('RGB')
		if self.transform:
			image = self.transform(image)
		#sample = {'image': image, 'label': label}
		return image, label
		
	def __len__(self):
		return len(self.frame)

def getTrainData(csv_file,batch_size=64):
	
	stats = {'mean': [0.3931, 0.3785, 0.3606],
			 'std': [0.1965, 0.1813, 0.1779]}

	transformed_training = custDataset(csv_file=csv_file,
											transform=transforms.Compose([
											transforms.Resize(224),
											transforms.CenterCrop(224),
											transforms.RandomHorizontalFlip(),
											transforms.RandomRotation(10),
											transforms.ToTensor(),
											transforms.Normalize(stats['mean'],stats['std'])
										]))

	dataloader_training = DataLoader(transformed_training, batch_size,
									 shuffle=True, num_workers=4, pin_memory=False)

	return dataloader_training



def getTestData(csv_file,batch_size=64):

	stats = {'mean': [0.3931, 0.3785, 0.3606],
			 'std': [0.1965, 0.1813, 0.1779]}
	
	transformed_testing = custDataset(csv_file=csv_file,
										transform=transforms.Compose([
										transforms.Resize(224),
										transforms.CenterCrop(224),
										transforms.ToTensor(),
										transforms.Normalize(stats['mean'],stats['std'])
									]))

	dataloader_testing = DataLoader(transformed_testing, batch_size,
									shuffle=True, num_workers=4, pin_memory=False)

	return dataloader_testing