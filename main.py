from asyncio import protocols, streams
from cgi import test
from copy import deepcopy
from msvcrt import setmode
from multiprocessing.dummy import active_children
from subprocess import list2cmdline
from tkinter import Y
from turtle import pos
from tensorflow.keras import Model, layers, initializers
from unicodedata import bidirectional
import tensorflow as tf
from tensorflow import keras
import numpy as np
import jieba
import pandas as pd
from keras.layers import Dense,Input,Flatten,Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate
from keras.preprocessing.text import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import pandas as pd
import os
import numpy as np
import re
from keras.layers import LSTM,LayerNormalization
from keras.layers import GRU
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import Bidirectional
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from keras_self_attention import SeqSelfAttention
# from keras.utils import to_categorical
from keras.layers import CuDNNLSTM,CuDNNGRU
from keras import backend as K
from keras.layers import Input,Dense,Dropout,Conv2D,MaxPool2D,Flatten,GlobalAvgPool2D,concatenate,BatchNormalization,Activation,Add,ZeroPadding2D,Lambda
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Activation,LayerNormalization
import csv

from sklearn.metrics import matthews_corrcoef,roc_auc_score

import time
import datetime
# 使用cpu进行模型训练
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.list_physical_devices('GPU')

# auc_best = 0
# fpr_best = 0
# tpr_best = 0
best_acc = 0
x_train = None
y_train = None
x_test = None
y_test = None
# auc_best_list = {}
accuracy_best_list = {}

def machine_learning(number):
    # global auc_best, fpr_best, tpr_best
    global x_test, y_test, x_train, y_train
    global best_acc
    global accuracy_best_list
    # global auc_best_list 
    
    # train_data = pd.read_csv(r"C:\Users\小米笔记本pro\Desktop\RBP-31\02_train.csv",header = None)
    # test_data  = pd.read_csv(r"C:\Users\小米笔记本pro\Desktop\RBP-31\02_test.csv",header = None)
    file_num = '%d' % number
    train_filename = 'C:/Users/deeplearning/Desktop/MaskDNA-PGD-main/MaskDNA-PGD-main/data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv'
    test_filename = 'C:/Users/deeplearning/Desktop/MaskDNA-PGD-main/MaskDNA-PGD-main/data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv'



    # train_data = pd.read_csv(r"C:\Users\小米笔记本pro\Desktop\RBP-31\01_train.csv",header = None)
    # test_data  = pd.read_csv(r"C:\Users\小米笔记本pro\Desktop\RBP-31\01_test.csv",header = None)
    
    Test_label = []
    best_acc = 0
    # train_data = pd.read_csv(train_filename,header = None,sep = "\t")
    # test_data  = pd.read_csv(test_filename,header = None,sep = "\t")

    train_data = pd.read_csv(train_filename,header = None, sep = "\t")
    test_data  = pd.read_csv(test_filename,header = None, sep = "\t")

    
  

    
    pro_x_train = train_data[2][1:]
    y_train =  train_data[1][:]
    pos_train_len = len(pro_x_train)

    print(y_train)
    pro_x_test =  test_data[2][1:]
    y_test = test_data[1][:]

    

    



    

    # neg_train = train_data[1][24001:]
    # neg_train_len = len(neg_train)

    # pos_test  = test_data[1][1:8001]
    # pos_test_len  = len(pos_test)

    # neg_test  = test_data[1][8001:]
    # neg_test_len  = len(neg_test)

    
    # pro_x_train = pd.concat([pos_train,neg_train],ignore_index= True )
    # x_train_len = len(pro_x_train)
    # pro_x_test  = pd.concat([pos_test,neg_test],ignore_index= True )
    # x_test_len  = len(pro_x_test)
    pro_x_data  = pd.concat([pro_x_train,pro_x_test],ignore_index= True )
    pro_y_data  = pd.concat([y_train,y_test],ignore_index= True )

    # pn = pd.concat([pos,neg],ignore_index= True )
    # print(type(pn))
    # print("pn[:10] before ->")
    # print(pn[:10])
    # print("pn[-10:] before ->")
    # print(pn[-10:])

   
    for i in range(1, len(pro_y_data)):
        if(pro_y_data[i] == "1"):
            Test_label.append([1,0])
        elif(pro_y_data[i] == "0"):
            Test_label.append([0,1])
        else:
            print(i)
            

    # print(len(pro_y_data))
    # print(len(Test_label))
   


    K = 3
    str_array = []
    loopcnt = 0
    for i in pro_x_data:
        seq_str = str(i)
        seq_str = seq_str.strip('[]\'')
        t=0
        l=[]
        for index in range(len(seq_str)):
            t=seq_str[index:index+K]
            if (len(t))==K:
                l.append(t)
        str_array.append(l)


    




    # print(type(pn))
    # print("pn[:10] after ->")
    # print(pn[:10])
    # print("pn[-10:] after ->")
    # print(pn[-10:])

    # neglen = len(neg)
    # print("neglen:",neglen)
    # poslen = len(pos)
    # print("poslen:",poslen)

    



    tokenizer = Tokenizer(num_words = 30000)
    tokenizer.fit_on_texts(str_array)
    sequences = tokenizer.texts_to_sequences(str_array)
    sequences = pad_sequences(sequences,maxlen = 48,padding = "post")
    # sequences = pad_sequences(sequences,maxlen = 200)
    sequences = np.array(sequences)

    # str_array2 = []

    # for i in sequences:
    #     str_array1 = []
    #     for char in i:
    #         if char == "A":
    #             str_array1.append([1,0,0,0,1,1,1])
    #         elif char == "U":
    #             str_array1.append([0,0,0,1,0,1,0])
    #         elif char == "C":
    #             str_array1.append([0,1,0,0,0,0,1])
    #         elif char == "G":
    #             str_array1.append([0,0,1,0,1,0,0])
    #     str_array2.append(str_array1)

    # sequences = str_array2


    x_train,x_test = sequences[:pos_train_len],sequences[pos_train_len:]

    y_train,y_test = Test_label[:pos_train_len],Test_label[pos_train_len:]
    

    

    def create_masked_data(x_train, mask_percentage):
   
        # Validate mask_percentage range
        if mask_percentage < 0.0 or mask_percentage > 1.0:
            raise ValueError("mask_percentage must be between 0 and 1.")

        # Create a mask of the same shape as x_train
        mask = np.random.random(x_train.shape) < mask_percentage

        # Apply the mask to create the masked data
        masked_data = np.where(mask, x_train, 0.0)

        return masked_data

    # Example usage
    mask_percentage = 1  # Control the percentage of masked values to 50%
    x_train = create_masked_data(x_train, mask_percentage)
    # x_train = np.concatenate((masked_data, x_train),axis = 1)
    
    # x_test = np.concatenate((x_test, x_test),axis = 1)
    

    # tokenizer = Tokenizer(num_words = 5)
    # tokenizer.fit_on_texts(pn)
    # sequences = tokenizer.texts_to_sequences(pn)
    # sequences = pad_sequences(sequences,maxlen = 200,padding = "post")
    # sequences = np.array(sequences)
    # print(sequences.shape)
    # dict_text = tokenizer.word_index
    # print(sequences[-2])


    # # convert to one hot
    # sequences = to_categorical(sequences, num_classes=5)
    # print(sequences[-2])
    # print(type(sequences))
    # print(sequences.shape)

    # sequences = np.delete(sequences, 0, axis=2)
    # print(sequences[-2])
    # print(sequences.shape)



    
    # pos_train_labels = [[1,0] for _ in range(pos_train_len)]
    # neg_train_lables = [[0,1] for _ in range(neg_train_len)]
    # pos_test_labels  = [[1,0] for _ in range(pos_test_len)]
    # neg_test_lables  = [[0,1] for _ in range(neg_test_len)]
    # y_train = np.concatenate([pos_train_labels,neg_train_lables],0)
    # y_test  = np.concatenate([pos_test_labels,neg_test_lables],0)

    # pos_labels = [[1,0] for _ in range(poslen)]
    # neg_lables = [[0,1] for _ in range(neglen)]
    # y = np.concatenate([pos_labels,neg_lables],0)
    # print(y.shape)

 
    t = time.time()
    my_time = int(round(t * 1000)) % 2147483648
    print(my_time)
    np.random.seed(my_time)
    # np.random.seed(10)
    
    print(x_train[1:10])
    print(y_test[:110])


    y_train = np.array(y_train)
    y_test = np.array(y_test)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = y_train[shuffle_indices]

    sequence_input = Input(shape = (48
    ,))
    embedding_layer = Embedding(30000,
                                16,
                                input_length = 48)
    embedded_sequences = embedding_layer(sequence_input)

    
    #-----------------------------------------1
    # cnn1 = Conv1D(filters = 32,kernel_size = 24,activation = "relu")(embedded_sequences)
    # cnn1 = MaxPooling1D(pool_size = 5)(cnn1)
    # cnn1 = Conv1D(filters = 32,kernel_size = 24,activation = "relu")(cnn1)
    # cnn1 = MaxPooling1D(pool_size = 5)(cnn1)
    # cnn1  = Flatten()(cnn1)

    # cnn2 = Conv1D(filters = 32,kernel_size = 16,activation = "relu")(embedded_sequences)
    # cnn2 = MaxPooling1D(pool_size = 5)(cnn2)
    # cnn2 = Conv1D(filters = 32,kernel_size = 16,activation = "relu")(cnn2)
    # cnn2 = MaxPooling1D(pool_size = 5)(cnn2)
    # cnn2  = Flatten()(cnn2)
    
    # cnn3 = Conv1D(filters = 64,kernel_size = 10,activation = "relu")(embedded_sequences)
    # cnn3 = MaxPooling1D(pool_size = 5)(cnn3)
    # cnn3 = Conv1D(filters = 64,kernel_size =10,activation = "relu")(cnn3)
    # cnn3 = MaxPooling1D(pool_size = 5)(cnn3)
    # cnn3  = Flatten()(cnn3)
    
    # merge_fla  = concatenate([cnn3,cnn2,cnn1],axis = 1)
    # merge_fla = Dropout(0.5)(merge_fla)

    #-----------------------------------------2

    # downsample1 =  embedded_sequences[:,0::2,:]
    # downsample2 =  embedded_sequences[:,1::2,:]
    # embedded_sequences  = concatenate([downsample2,downsample1],axis = 2)

    stem = LayerNormalization(epsilon=1e-6)(embedded_sequences)
    stem = Dropout(0.2)(stem) 

    # stem = Conv1D(filters = 96,kernel_size = 8,padding="same",activation = "gelu")(embedded_sequences)
    # lstm = Bidirectional(CuDNNLSTM(16,return_sequences = True))(embedded_sequences)
    # lstm = layers.Activation("gelu")(lstm)
    # stem = concatenate([stem,lstm],axis = 2)
    # stem = BatchNormalization(epsilon=1e-6)(stem)
    # stem = Dropout(0.5)(stem)
    

    # downsample1 =  stem[:,0::2,:]
    # downsample2 =  stem[:,1::2,:]
    # stem  = concatenate([downsample2,downsample1],axis = 2)

    MLP_1  = Dense(int(16 * 2.0), name="Dense_0",kernel_initializer= initializers.GlorotUniform(),
    bias_initializer = initializers.RandomNormal(stddev=1e-6))(stem)
    Activation_1 = layers.Activation("gelu")(MLP_1)
    Activation_1 = LayerNormalization(epsilon=1e-6)(Activation_1)
    Dropout_2 = Dropout(0.2)(Activation_1)

    MLP_2  = Dense(int(16* 1.0), name="Dense_1",kernel_initializer= initializers.GlorotUniform(),
    bias_initializer = initializers.RandomNormal(stddev=1e-6))(Dropout_2)
    MLP_2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(MLP_2,stem)
    stem = stem + MLP_2
    stem = Dropout(0.2)(stem)
    


    lstm = Bidirectional(CuDNNLSTM(16,return_sequences = True))(stem)
    lstm = layers.Activation("gelu")(lstm)
    lstm = LayerNormalization(epsilon=1e-6)(lstm)
    lstm = Dropout(0.5)(lstm)

    lstm1 = Bidirectional(CuDNNLSTM(16,return_sequences = True))(lstm)
    lstm1 = layers.Activation("gelu")(lstm1)
    lstm1 = LayerNormalization(epsilon=1e-6)(lstm1)
    lstm1 = Dropout(0.2)(lstm1)
    lstm1 = lstm1 + lstm 
    
    lstm2 = Bidirectional(CuDNNLSTM(16,return_sequences = True))(lstm1)
    lstm2 = layers.Activation("gelu")(lstm2)
    lstm2 = LayerNormalization(epsilon=1e-6)(lstm2)
    lstm2 = Dropout(0.2)(lstm2)

    lstm2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(lstm2,lstm)
    lstm2 = lstm2 + lstm

    # lstm = Flatten()(lstm)


    cnn1_24 = Conv1D(filters = 32,kernel_size = 9,strides=1,activation = "gelu",padding='causal')(stem) 
    cnn1_16 = Conv1D(filters = 64,kernel_size = 8,strides=1,activation = "gelu",padding='causal')(stem)
    cnn1_10 = Conv1D(filters = 32,kernel_size = 5,strides=1,activation = "gelu",padding='causal')(stem)
    merge_fla_1  = concatenate([cnn1_24,cnn1_16,cnn1_10],axis = 2)
    merge_fla_1 = LayerNormalization(epsilon=1e-6)(merge_fla_1)
    merge_fla_1 = Dropout(0.2)(merge_fla_1)
    
    
    cnn2_24 = Conv1D(filters = 32,kernel_size = 9,strides=1,activation = "gelu",padding='causal')(merge_fla_1)
    cnn2_16 = Conv1D(filters = 64,kernel_size = 8,strides=1,activation = "gelu",padding='causal')(merge_fla_1)
    cnn2_10 = Conv1D(filters = 32,kernel_size = 5,strides=1,activation = "gelu",padding='causal')(merge_fla_1)

    merge_fla_2  = concatenate([cnn2_24,cnn2_16,cnn2_10],axis = 2)
    merge_fla_2 = merge_fla_2 + merge_fla_1
    merge_fla_2 = LayerNormalization(epsilon=1e-6)(merge_fla_2)
    merge_fla_2 = Dropout(0.2)(merge_fla_2)
    

    cnn3_24 = Conv1D(filters = 32,kernel_size = 9,strides=1,activation = "gelu",padding='causal')(merge_fla_2)
    cnn3_16 = Conv1D(filters = 64,kernel_size = 8,strides=1,activation = "gelu",padding='causal')(merge_fla_2)
    cnn3_10 = Conv1D(filters = 32,kernel_size = 5,strides=1,activation = "gelu",padding='causal')(merge_fla_2)
    
    merge_fla_3  = concatenate([cnn3_24,cnn3_16,cnn3_10],axis = 2)
    merge_fla_3 = merge_fla_2 + merge_fla_3
    merge_fla_3 = LayerNormalization(epsilon=1e-6)(merge_fla_3)
    merge_fla_3 = Dropout(0.2)(merge_fla_3) 
    
    cnn4_24 = Conv1D(filters = 32,kernel_size = 9,strides=1,activation = "gelu",padding='causal')(merge_fla_3)
    cnn4_16 = Conv1D(filters = 64,kernel_size = 8,strides=1,activation = "gelu",padding='causal')(merge_fla_3)
    cnn4_10 = Conv1D(filters = 32,kernel_size = 5,strides=1,activation = "gelu",padding='causal')(merge_fla_3)
    
    merge_fla_4  = concatenate([cnn4_24,cnn4_16,cnn4_10],axis = 2)
    merge_fla_4 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merge_fla_1,merge_fla_4)
    merge_fla_4 = merge_fla_1 + merge_fla_4
    merge_fla_4 = LayerNormalization(epsilon=1e-6)(merge_fla_4) 
    merge_fla_4 = Dropout(0.2)(merge_fla_4) 
    lstm  = concatenate([merge_fla_4,lstm2])


    lstm  = Flatten()(lstm)
    # merge_fla = Dropout(0.2)(lstm)




    
    merge = Dense(200,activation = "sigmoid")(lstm)
    merge = Dropout(0.5)(merge)
    

  


    # merge = Dense(16,activation = "sigmoid")(merge)
    # # merge = Dropout(0.5)(merge)


    preds = Dense(2,activation = "softmax")(merge)
    model = Model(sequence_input,preds)

    model.summary()


    # model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
    # model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=[auc])
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])


    # my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

    # class LossHistory(Callback):
    #     def on_epoch_end(self, epoch, logs={}):
    #         global best_acc
    #         epoch_pred = self.model.predict(x_test)
    #         for i in range(len(epoch_pred)):
    #                 max_value=max(epoch_pred[i])
    #                 for j in range(len(epoch_pred[i])):
    #                     if max_value==epoch_pred[i][j]:
    #                         epoch_pred[i][j]=1
    #                     else:
    #                         epoch_pred[i][j]=0
    #         print()
    #         print("epoch[",epoch+1,"].pred:\n", classification_report(y_test, epoch_pred, digits=5))
    #         my_valacc = logs.get('val_accuracy')
    #         print()
    #         if my_valacc > best_acc:
    #             best_acc = my_valacc
    #         print("epoch[",epoch+1,"].val_accuracy:", my_valacc)
    #         print("epoch[",epoch+1,"].best_accuracy:", best_acc)
    #         print()
    #         print()

    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)
    # history = LossHistory()
    # history = AUCHistory()

    class AUCMCCCallback(Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data
            self.last_epoch_logs = {}
        
        def on_epoch_end(self, epoch, logs={}):
            # 保存当前 epoch 的日志
            self.last_epoch_logs = logs.copy()

        def on_epoch_begin(self, epoch, logs={}):
            if epoch < 1:
                return
            x_val, y_val = self.validation_data
            y_pred = self.model.predict(x_val)
            # y_pred = self.model.predict(x_val, batch_size=None, verbose=1)
            auc = roc_auc_score(y_val, y_pred)
            mcc = matthews_corrcoef(y_val.argmax(axis=1), y_pred.argmax(axis=1))
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1))
            print(f'\nValidation AUC: {auc:.4f} - MCC: {mcc:.4f} - ACC: {accuracy:.4f}')
            global best_acc
            epoch_pred = self.model.predict(x_test)
            for i in range(len(epoch_pred)):
                    max_value=max(epoch_pred[i])
                    for j in range(len(epoch_pred[i])):
                        if max_value==epoch_pred[i][j]:
                            epoch_pred[i][j]=1
                        else:
                            epoch_pred[i][j]=0
            # print()
            # print("epoch[",epoch+1,"].pred:\n", classification_report(y_test, epoch_pred, digits=5))
            # my_valacc = logs.get('val_accuracy')
            #
            my_valacc = self.last_epoch_logs.get('val_accuracy', 0)
            print()
            if my_valacc > best_acc:
                best_acc = my_valacc
            print("epoch[",epoch,"].val_accuracy:", my_valacc)
            print("epoch[",epoch,"].best_accuracy:", best_acc)
            print()
            print()
            

    auc_mcc_callback = AUCMCCCallback(validation_data=(x_test, y_test))
    model.fit(x_train,y_train,
            batch_size = 30,
            # epochs = 100,
            epochs = 100,
            validation_data = (x_test,y_test),
            callbacks=[auc_mcc_callback])
    
    # y_pred = model.predict(x_test)
    print()
    accuracy_best_list[number] = best_acc

    for i in accuracy_best_list.keys():
        print("best_acc[",i,"] = ", accuracy_best_list[i])
    avg = float(sum(accuracy_best_list.values())) / len(accuracy_best_list)
    print()
    print("best_acc[avg] = ", avg)
    print()
    print()
    
for i in range(0,1):
    machine_learning(i)