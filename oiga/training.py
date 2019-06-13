from .cnn_model import create_model
from .data_gathering import load_dataset_from_pckl
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np

def train(train_data, train_labels_one_hot, test_data, test_labels_one_hot):
    model1 = create_model((1,65,99), 2)
    print('model created')
    batch_size = 50
    epochs = 15

    adam = Adam(lr=0.00001, beta_1=0.8, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    model1.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                 validation_data=(test_data, test_labels_one_hot))

    model1.save_weights('model.h5')
    
    print("Here you got some results")
    print(model1.evaluate(test_data, test_labels_one_hot))

    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    
    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    plt.show()


print('imports completed')
trn_x, trn_Y, tst_x, tst_Y = load_dataset_from_pckl('dataset.p')
s = []
print(len(trn_x))
print(len(trn_Y))
# print(tst_x)
# print(tst_Y)
for i in trn_x:
    s.append(i.shape)

trn_x = trn_x.reshape(trn_x.shape[0], 1, trn_x.shape[1], trn_x.shape[2]).astype('float32')
tst_x = tst_x.reshape(tst_x.shape[0], 1, tst_x.shape[1], tst_x.shape[2]).astype('float32')

print('Training data')
print(trn_x.shape)
print('Training label')
print(trn_Y.shape)
print('Test data')
print(tst_x.shape)
print('Test label')
print(tst_Y.shape)

np.random.seed(13)
np.random.shuffle(trn_x)
np.random.seed(13)
np.random.shuffle(trn_Y)
np.random.seed(13)
np.random.shuffle(tst_x)
np.random.seed(13)
np.random.shuffle(tst_Y)

trn_x = trn_x/255.
trn_x[np.where(trn_x==np.inf)]=1.
trn_x[np.where(trn_x==np.nan)]=0.
trn_x[np.where(trn_x==-np.inf)]=-1
tst_x = tst_x/255.
tst_x[np.where(tst_x==np.inf)]=1.
tst_x[np.where(tst_x==np.nan)]=0.
tst_x[np.where(tst_x==-np.inf)]=-1

train(trn_x, trn_Y, tst_x, tst_Y)