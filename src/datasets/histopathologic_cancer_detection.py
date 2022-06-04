import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class cancer_dataset():
  api_token = {"username":"youssefhassanein","key":"50f1ebf87c572bd3f25652e03343b2ce"}
  df_train = None
  df_val = None
  train_gen = None
  val_gen = None
  def __init__(self, basePath) -> None:
    df_train = pd.read_csv(f'{basePath}train_labels.csv', dtype=str)
  
  def load_img(path, color=True, size=None):
    img = cv2.imread(path) if color else cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None: return None
    
    if size is not None and len(size) == 2:
      img = cv2.resize(img, size)
    return img
  
  def displaySamples(trainPath):
    fig = plt.figure(figsize=(15, 11))
    fig.tight_layout()

    for i, val in enumerate(df_train.values[:24]):
      id, lab = val[0], val[1]
      ax = fig.add_subplot(4, 6, i+1)
      ax.set_title('Label: ' + lab)
      img = load_img(f'{trainPath}{id}{IMG_EXT}')
    plt.imshow(img)

  # Apply functions
  def add_ext(img_id):
    return img_id + ".tif"

  def remove_ext(img):
    return img.split('.')[0]

  def balanceDataset():
    df_train, df_val = train_test_split(df_train, test_size=0.20, random_state=42, stratify=df_train['label'])
  
  def initiateDataTensor(trainDirPath, batchSize):
    data_gen = ImageDataGenerator(rescale=1.0/255)
    train_gen = data_gen.flow_from_dataframe(dataframe=df_train,
                                        directory=trainDirPath,
                                        x_col='id',
                                        y_col='label',
                                        batch_size=batchSize,
                                        seed=42,
                                        shuffle=True,
                                        class_mode='binary',
                                        target_size=(96, 96))

    val_gen = data_gen.flow_from_dataframe(dataframe=df_val,
                                        directory=trainDirPath,
                                        x_col='id',
                                        y_col='label',
                                        batch_size=batchSize,
                                        seed=42,
                                        shuffle=True,
                                        class_mode='binary',
                                        target_size=(96, 96))
