import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class dataset():
  api_token = {"username":"youssefhassanein","key":"50f1ebf87c572bd3f25652e03343b2ce"}
  df_train = None
  df_val = None
  train_gen = None
  val_gen = None
  def __init__(self, basePath) -> None:
    self.df_train = pd.read_csv(f'{basePath}train_labels.csv', dtype=str)
  
  def load_img(self, path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None: return None
    
    return img
  
  def displaySamples(self, trainDirPath):
    fig = plt.figure(figsize=(15, 11))
    fig.tight_layout()

    for i, val in enumerate(self.df_train.values[:24]):
      id, lab = val[0], val[1]
      ax = fig.add_subplot(4, 6, i+1)
      ax.set_title('Label: ' + lab)
      img = self.load_img(f'{trainDirPath}{id}.tif')
      plt.imshow(img)
    plt.show()

  # Apply functions
  def add_ext(self, img_id):
    return img_id + ".tif"

  def remove_ext(self, img):
    return img.split('.')[0]

  def balanceDataset(self):
    self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.20, random_state=42, stratify=self.df_train['label'])
  
  def initiateDataTensor(self, trainDirPath, batchSize, size):
    self.df_train['id'] = self.df_train['id'].apply(self.add_ext)
    self.df_val['id'] = self.df_val['id'].apply(self.add_ext)
    data_gen = ImageDataGenerator(rescale=1.0/255)
    self.train_gen = data_gen.flow_from_dataframe(dataframe=self.df_train[:size],
                                        directory=trainDirPath,
                                        x_col='id',
                                        y_col='label',
                                        batch_size=batchSize,
                                        seed=42,
                                        shuffle=True,
                                        class_mode='binary',
                                        target_size=(96, 96))

    self.val_gen = data_gen.flow_from_dataframe(dataframe=self.df_val[:int(size*0.2)],
                                        directory=trainDirPath,
                                        x_col='id',
                                        y_col='label',
                                        batch_size=batchSize,
                                        seed=42,
                                        shuffle=True,
                                        class_mode='binary',
                                        target_size=(96, 96))
    self.df_train['id'] = self.df_train['id'].apply(self.remove_ext)
    self.df_val['id'] = self.df_val['id'].apply(self.remove_ext)
