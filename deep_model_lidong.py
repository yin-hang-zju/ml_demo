# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, BatchNormalization
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn import preprocessing


class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print '\rroc-auc_train: %s - roc-auc_val: %s ' % (str(round(roc_train,4)),str(round(roc_val,4)))
        end_str = 100 *' '+'\n'
        print end_str
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


date = '2020-02-24'
app_dim = 281


def handle_file():
    file_path = "/home/hdp-growth/lidong/personal_push/data/push_train_feature_matrix_%s_200.csv" % str(date)
    # load the dataset


    # 将整型变为float
    dataframe = pd.read_csv(file_path, header=0, na_filter=False, error_bad_lines=False)
    #dataframe.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    return dataset


def train_model(data, rate=0.8):
    #train_x = data[:1500000, 2:]

    split_index = int(data.shape[0] * rate)
    train_x_wide = np.hstack((data[:split_index, 2:9], data[:split_index, 209:]))
    train_x_wide = preprocessing.scale(train_x_wide)
    train_x_deep = data[:split_index, 9:209]
    train_x = np.hstack((train_x_wide, train_x_deep))
    train_y = data[:split_index, 1]
    #test_x = data[1500000:, 2:]
    test_x = data[split_index:, 9:209]
    test_x_wide = np.hstack((data[split_index:, 2:9], data[split_index:, 209:]))
    test_x_wide = preprocessing.scale(test_x_wide)
    test_x_deep = data[split_index:, 9:209]
    test_x = np.hstack((test_x_wide, test_x_deep))
    test_y = data[split_index:, 1]

    model = Sequential()
    model.add(Dense(64, input_dim=app_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(24))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(6))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    roc = RocCallback(training_data=(train_x, train_y),
                      validation_data=(test_x, test_y))
    #history = model.fit(train_x, train_y, batch_size=1000, epochs=400)
    history = model.fit(train_x, train_y, validation_data = (test_x, test_y), callbacks = [roc],batch_size=2000, epochs=40)

    model.save("/home/hdp-growth/lidong/personal_push/data/model/deep_model_%s" % date)
    losses, accuracy = model.evaluate(test_x, test_y)

    print losses
    print accuracy


if __name__ == "__main__":
    data = handle_file()
    train_model(data)