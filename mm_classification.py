import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2, VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, concatenate, GlobalAveragePooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from weights_transfer import load_weights
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.0003
    if epoch > 800:
        lr *= 0.5e-3
    elif epoch > 600:
        lr *= 1e-3
    elif epoch > 400:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# 图像数据读取函数
def get_images(path, size=(250,250,3), normalized=True):
    """
    获得模型学习所需要的数据；
    其中图像格式(num_images, weight, height)
    标注格式(num_images, weight, height)，像素值为0/1
    注：训练数据目录结构如下
    path/
            0/   - neg
            1/   - pos
    """

    files_neg = os.listdir(os.path.join(path, '0'))
    files_neg.sort()

    files_pos = os.listdir(os.path.join(path, '1'))
    files_pos.sort()

    images = np.zeros([len(files_neg)+len(files_pos),size[0],size[1],size[2]])
    for i, file in enumerate(files_neg):
        img = cv2.imread(os.path.join(path, '0', file))
        img = cv2.resize(img, (size[0], size[1]), cv2.INTER_AREA)
        if normalized:
            images[i] = img/255
        else:
            images[i] = img

    for i, file in enumerate(files_pos):
        img = cv2.imread(os.path.join(path, '1', file))
        img = cv2.resize(img, (size[0], size[1]), cv2.INTER_AREA)
        if normalized:
            images[i+len(files_neg)] = img/255
        else:
            images[i+len(files_neg)] = img

    return images


# DNN模型用于临床数据
def create_mlp(input_dim, nlogits=1536):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation="relu", name="dense_clinic_1"))
    model.add(Dense(nlogits, activation="relu", name="dense_clinic_2"))
    model.add(Reshape((1, nlogits), name="reshape_clinic_1"))

    return model

# 临床数据分支
input_dim = 3
nlogits = 1536
mlp = create_mlp(input_dim,nlogits)
#mlp.summary()

# cnn模型特征输入分支
XSIZE,YSIZE,ZSIZE = 250,250,3

# transfer learning setting
#   0 - no initial weights
#   1 - imagenet pretrained weights
#   2 - transfer learning weights from SMC-net
#
transfer_learning_style = 2  

# gray-image input
if transfer_learning_style==0:
    ir2_gray = InceptionResNetV2(include_top=False,input_shape=(XSIZE,YSIZE,ZSIZE))
elif transfer_learning_style==1:
    gray_weight_file = "./weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ir2_gray = InceptionResNetV2(include_top=False, weights=gray_weight_file, input_shape=(XSIZE, YSIZE, ZSIZE))
else:    
    ir2_gray = InceptionResNetV2(include_top=False,input_shape=(XSIZE,YSIZE,ZSIZE))
    gray_weight_file = "./saved_models/InceptionResNetV2_GRAY.554.h5"
    gray_weight = load_weights(gray_weight_file)
    ir2_gray.set_weights(gray_weight)

model_gray = GlobalAveragePooling2D(name='GlobalAverage2D_gray')(ir2_gray.output)
model_gray = Reshape((1,-1),name="reshape_all_gray")(model_gray)
cnn_gray = Model(inputs=ir2_gray.input, outputs=model_gray, name="model_gray")
for layer in ir2_gray.layers:
    layer.trainable = True
    layer._name = layer._name + str("_gray")

# color-image input
if transfer_learning_style==0:
    ir2_color = InceptionResNetV2(include_top=False,input_shape=(XSIZE,YSIZE,ZSIZE))
elif transfer_learning_style==1:
    color_weight_file = "./weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ir2_color = InceptionResNetV2(include_top=False, weights=gray_weight_file, input_shape=(XSIZE, YSIZE, ZSIZE))
else:    
    ir2_color = InceptionResNetV2(include_top=False,input_shape=(XSIZE,YSIZE,ZSIZE))
    color_weight_file = "./saved_models/InceptionResNetV2_COLOR.549.h5"
    color_weight = load_weights(color_weight_file)
    ir2_color.set_weights(color_weight)

model_color = GlobalAveragePooling2D(name='GlobalAverage2D_color')(ir2_color.output)
model_color = Reshape((1,-1),name="reshape_all_color")(model_color)
cnn_color = Model(inputs=ir2_color.input, outputs=model_color,name="model_color")
for layer in ir2_color.layers:
    layer.trainable = True
    layer._name = layer._name + str("_color")

# 模型输入的融合
combinedInput = concatenate(axis=1, inputs=[mlp.output, cnn_gray.output, cnn_color.output], name="concatenate_all")

# LSTM模型 
outputs = LSTM(128,dropout=0.25,input_shape=combinedInput.shape,name="LSTM_all")(combinedInput)
outputs = Dense(128, activation="relu", name="dense_output_1")(outputs)
outputs = Dropout(0.5, name="dropout_output_1")(outputs)
outputs = Dense(32, activation="relu", name="dense_output_2")(outputs)
outputs = Dense(1, activation="sigmoid", name="dense_output_3")(outputs)
model = Model(inputs=[mlp.input, cnn_gray.input, cnn_color.input], outputs=outputs)
#model.summary()

# 模型编译
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003), metrics=['acc'])

# 获得临床数据
pos = pd.read_excel('./data/MultiModality/training/clinic/positive.xlsx')
neg = pd.read_excel('./data/MultiModality/training/clinic/negative.xlsx')

# 临床数据预处理
#
#   男 - 1
#   女 - 0
#
neg['sex'][neg['sex'] == '女'] = 0
neg['sex'][neg['sex'] == '男'] = 1
pos['sex'][pos['sex'] == '女'] = 0
pos['sex'][pos['sex'] == '男'] = 1

# 临床数据整合为特征和标签
posv = pos.values.astype(np.float32)
negv = neg.values.astype(np.float32)
x = np.concatenate((negv, posv))
y_train_clinic = np.concatenate( (np.zeros(len(negv)), np.ones(len(posv))) )
x_train_clinic = StandardScaler().fit_transform(x)
print("用于训练的临床数据为", len(x_train_clinic), " 标签为", len(y_train_clinic))

# 黑白图像数据
x_train_gray = get_images("./data/MultiModality/training/B-mode/")
x_train_color = get_images("./data/MultiModality/training/CDFI/")

# 用于训练的数据大小
print("用于训练的灰度图像数据为", x_train_gray.shape)
print("用于训练的彩色图像数据为", x_train_color.shape)

# 训练轮数
epochs = 1000
batch_size = 32

# 保存模型权重
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'MMC-Net.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True)
callbacks = [checkpoint]

history = model.fit([x_train_clinic, x_train_gray, x_train_color],y_train_clinic,batch_size=batch_size,
                    validation_split=0.15, epochs=epochs, callbacks=callbacks, shuffle=True)

history_saved = True
if history_saved:
   history_file = os.path.join('./history', 'mm_history-epoch-'+str(epochs)+'.dat')
   with open(history_file,'wb') as f:
       pickle.dump(history.history, f)
   print("Network training history successfully saved!")