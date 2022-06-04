import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

class model1:
  instance = None
  
  def __init__(self) -> None:
    if os.path.exists('model1.h5'):
      print('Loading model1...')
      instance = load_model('model1.h5')
    else:
      print('Building and compiling model1...')
      instance = self.build_compile_model1()
  
  def build_compile_model1():
      model = Sequential()

      model.add(Conv2D(filters=256, kernel_size=(3,3), input_shape=((96, 96), 3), activation='relu'))
      model.add(MaxPooling2D(pool_size=(2,2)))

      model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
      model.add(MaxPooling2D(pool_size=(2,2)))

      model.add(Flatten())

      model.add(Dense(64, activation='relu'))

      model.add(Dense(1))
      model.add(Activation('sigmoid'))

      model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

      return model

  def summary():
    if instance != None:
      return instance.summary()
  
  def fit(train_gen, val_gen):
    if instance != None:
      # It alters the learning rate based on metrics in each epoch
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                    verbose=1, mode='min', min_lr=0.00001)
      callbacks_list = [reduce_lr]
      return instance.fit(train_gen, 
                      epochs=10, 
                      validation_data=val_gen, 
                      callbacks=callbacks_list)
  
