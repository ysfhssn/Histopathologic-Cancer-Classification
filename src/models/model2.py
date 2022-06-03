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
    if os.path.exists('model2.h5'):
      print('Loading model1...')
      instance = load_model('model2.h5')
    else:
      print('Building and compiling model1...')
      instance = self.build_compile_model2()
  
  def build_compile_model2():
      kernel_size = (3,3)
      pool_size= (2,2)
      first_filters = 32
      second_filters = 64
      third_filters = 128
      dropout_conv = 0.3
      dropout_dense = 0.3

      # Model Structure
      model = Sequential()
      model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape=((96,96), 3)))
      model.add(Conv2D(first_filters, kernel_size, activation='relu'))
      model.add(Conv2D(first_filters, kernel_size, activation='relu'))
      model.add(MaxPooling2D(pool_size=pool_size)) 
      model.add(Dropout(dropout_conv))

      model.add(Conv2D(second_filters, kernel_size, activation='relu'))
      model.add(Conv2D(second_filters, kernel_size, activation='relu'))
      model.add(Conv2D(second_filters, kernel_size, activation='relu'))
      model.add(MaxPooling2D(pool_size=pool_size))
      model.add(Dropout(dropout_conv))

      model.add(Conv2D(third_filters, kernel_size, activation='relu'))
      model.add(Conv2D(third_filters, kernel_size, activation='relu'))
      model.add(Conv2D(third_filters, kernel_size, activation='relu'))
      model.add(MaxPooling2D(pool_size=pool_size))
      model.add(Dropout(dropout_conv))

      model.add(Flatten())
      model.add(Dense(256, activation="relu"))
      model.add(Dropout(dropout_dense))
      model.add(Dense(1, activation="sigmoid"))

      model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy'])

      return model
  
  def summary():
    if instance != None:
      return instance.summary()
  
  def fit(train_gen, val_gen):
    # It alters the learning rate based on metrics in each epoch
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                   verbose=1, mode='min', min_lr=0.00001)                                                  
    callbacks_list = [reduce_lr]

    # Fitting the model
    history = instance.fit(train_gen, 
                    epochs=10, 
                    validation_data=val_gen, 
                    callbacks=callbacks_list)