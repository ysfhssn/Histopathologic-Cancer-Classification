import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class model1:
  instance = None
  
  def __init__(self) -> None:
    if os.path.exists('model1.h5'):
      print('Loading model1...')
      self.instance = load_model('model1.h5')
    else:
      print('Building and compiling model1...')
      self.instance = self.build_compile_model1()
  
  @staticmethod
  def build_compile_model1():
      model = Sequential()

      model.add(Conv2D(filters=256, kernel_size=(3,3), input_shape=(96, 96, 3), activation='relu'))
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

  def summary(self):
    if self.instance != None:
      return self.instance.summary()
  
  def fit(self,train_gen, val_gen, epochs):
    if self.instance != None:
      # It alters the learning rate based on metrics in each epoch
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                    verbose=1, mode='min', min_lr=0.00001)
      callbacks_list = [reduce_lr]
      return self.instance.fit(train_gen, 
                      epochs=epochs, 
                      validation_data=val_gen, 
                      callbacks=callbacks_list)
  
