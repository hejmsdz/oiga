from keras.models import Sequential
from keras.layers import Dense,Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
  
def create_model(input_shape, nClasses):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Dropout(0.25))
  
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Dropout(0.25))
  
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nClasses, activation='softmax'))
  
    return model