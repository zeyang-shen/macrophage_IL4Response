from __future__ import print_function
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import random

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import tensorflow.keras.backend as K

batch_size = 512
num_classes = 1
epochs = 200
n_channels = 4
seq_size = 300
input_shape = (seq_size, n_channels)

def data_prep(path, label, seq_length):
    data_list = []
    for record in SeqIO.parse(path, "fasta"):
        info_tag = record.id
        chromID = info_tag.split("_")[0]
        chromSeq = (str(record.seq)).upper()
        if len(chromSeq) < seq_length:
            continue
        offset_start = int((len(chromSeq)-seq_length)/2.0)
        chromSeq_trimmed = chromSeq[offset_start : offset_start+seq_length]
        data_list.append((chromSeq_trimmed, label, chromID, info_tag))
    return data_list

def create_dataset(dataset, train, valid, test):
    for data_point in dataset:
        chromID = data_point[2]
        if chromID == "chr8":
            valid.append(data_point)
        elif chromID == "chr9":
            test.append(data_point)
        else:
            train.append(data_point)

def dataset2onehot(dataset, shuffle=True):
    nucleotides = ["A", "T", "C", "G"]
    def seq2onehot(seq):
        onehot_list = []
        for nuc in seq:
            if nuc == "N":
                onehot = [0.25 for _ in range(len(nucleotides))]
                onehot_list.append(onehot)
            else:
                onehot = [0 for _ in range(len(nucleotides))]
                onehot[nucleotides.index(nuc)] = 1
                onehot_list.append(onehot)
        return onehot_list
    
    def rc(seq):
        return str((Seq(seq)).reverse_complement())
    
    onehot_dataset = []
    for (seq, label, chromID, tag_info) in dataset:
        onehot_dataset.append((seq2onehot(seq), label, (tag_info, "+")))
        onehot_dataset.append((seq2onehot(rc(seq)), label, (tag_info, "-")))
    
    if shuffle:
        random.shuffle(onehot_dataset)
    
    x_list, y_list, info_list = [], [], [] 
    for (x, y, info) in onehot_dataset:
        x_list.append(x)
        y_list.append(y)
        info_list.append(info)
    return np.array(x_list), np.array(y_list), info_list

def load_from_tf(name):
    return tf.train.load_variable("./data/model", name=name) #load DeepSEA architecture and weights

def construct_model():
    # model structure
    model = Sequential()
    # conv1
    model.add(Conv1D(320, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=5e-6),
        bias_initializer=keras.initializers.Constant(value=0),
        input_shape=input_shape, name='conv1'))
    model.add(BatchNormalization(name = "bn1"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(Dropout(0.2))

    # conv2
    model.add(Conv1D(480, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=5e-6),
        bias_initializer=keras.initializers.Constant(value=0),
        name='conv2'))
    model.add(BatchNormalization(name = "bn2"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(Dropout(0.2))

    # conv3
    model.add(Conv1D(960, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=5e-6),
        bias_initializer=keras.initializers.Constant(value=0),
        name='conv3'))
    model.add(BatchNormalization(name = "bn3"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # fc1
    model.add(Dense(925,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=5e-6),
        bias_initializer=keras.initializers.Constant(value=0)))
    model.add(Activation('relu'))

    # output
    model.add(Dense(num_classes,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=5e-6),
        bias_initializer=keras.initializers.Constant(value=0)))
    model.add(Activation('sigmoid'))
    # end
    
    # load pre-trained model
    model.get_layer("conv1").set_weights(
        [load_from_tf('conv1/weights/ExponentialMovingAverage'),
        load_from_tf('conv1/biases/ExponentialMovingAverage')])
    model.get_layer("conv2").set_weights(
        [load_from_tf('conv2/weights/ExponentialMovingAverage'),
        load_from_tf('conv2/biases/ExponentialMovingAverage')])
    model.get_layer("conv3").set_weights(
        [load_from_tf('conv3/weights/ExponentialMovingAverage'),
        load_from_tf('conv3/biases/ExponentialMovingAverage')])
    model.get_layer("bn1").set_weights(
        [load_from_tf('conv1/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv1/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn1").weights[2].shape),
        np.zeros(model.get_layer("bn1").weights[3].shape)])
    model.get_layer("bn2").set_weights(
        [load_from_tf('conv2/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv2/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn2").weights[2].shape),
        np.zeros(model.get_layer("bn2").weights[3].shape)])
    model.get_layer("bn3").set_weights(
        [load_from_tf('conv3/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv3/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn3").weights[2].shape),
        np.zeros(model.get_layer("bn3").weights[3].shape)])
    
    return model

def run(pos_data_path, neg_data_path, save_path):
    pos_data = data_prep(pos_data_path, 1, seq_size)
    neg_data = data_prep(neg_data_path, 0, seq_size)

    train_raw, valid_raw, test_raw = [], [], []
    create_dataset(pos_data, train_raw, valid_raw, test_raw)
    create_dataset(neg_data, train_raw, valid_raw, test_raw)
    x_train, y_train, info_train = dataset2onehot(train_raw)
    x_valid, y_valid, info_valid = dataset2onehot(valid_raw)
    x_test, y_test, info_test = dataset2onehot(test_raw)
    x_vis, y_vis, info_vis = dataset2onehot(pos_data, False)

    model = construct_model()
    
    # model training
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            metrics=['binary_accuracy'])

    keras_model_save = "%s_model" %(save_path)
    modelCheckpoint = keras.callbacks.ModelCheckpoint(
        keras_model_save, monitor='val_loss', save_best_only=True)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.9, patience=5, min_delta=0.0001, min_lr=1e-5)
    history = keras.callbacks.History()
    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[earlyStopping, history, modelCheckpoint, reduceLROnPlateau],
              use_multiprocessing=True)

    # save model
    best_model = keras.models.load_model(keras_model_save)
    keras_model_weights = "%s_modelWeights.h5" %(save_path)
    keras_model_json = "%s_modelJson_tmp.json" %(save_path)
    json_string = best_model.to_json() # architecture
    with open(keras_model_json, 'w') as json_file:
        json_file.write(json_string)
    best_model.save_weights(keras_model_weights) # weights
    
    history_path = "%s_history.txt" %(save_path)
    with open(history_path, 'w') as ofile:
        for name in history.history:
            ofile.write("%s\n" %(name))
            for val in history.history[name]:
                ofile.write("%f\t" %(val))
            ofile.write("\n")

# exp 1
save_path = "./tmp/"
pos_data_path = "./data/model1_IL4Enhancer/C57Bl6_IL4Enhancers.positive.fa"
neg_data_path = "./data/model1_IL4Enhancer/C57Bl6_IL4Enhancers.negative.fa"
run(pos_data_path, neg_data_path, save_path)

# exp 2
save_path = "./tmp/"
pos_data_path = "./data/model2_IL4InducedEnhancer/Strains_IL4InducedEnhancers.positive.fa"
neg_data_path = "./data/model2_IL4InducedEnhancer/Strains_IL4InducedEnhancers.negative.fa"
run(pos_data_path, neg_data_path, save_path)

# exp 3
save_path = "./tmp/"
pos_data_path = "./data/model3_IL4induced_vs_noninduced/Strains_IL4InducedEnhancers.vs.C57_basalNoninducedEnhancers.positive.fa"
neg_data_path = "./data/model3_IL4induced_vs_noninduced/Strains_IL4InducedEnhancers.vs.C57_basalNoninducedEnhancers.negative.fa"
run(pos_data_path, neg_data_path, save_path)
