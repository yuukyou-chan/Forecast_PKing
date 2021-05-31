import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


yiliao = pd.read_csv('./SH600519.csv')
#print(yiliao)
training_set = yiliao.iloc[0:2426 - 300, 2:3].values  #前（1975-300）天的净值作为训练集，表格从0开始计数，2:3是提取[1:2）列，前闭后开，故提取出第二列净值
test_set = yiliao.iloc[2426 - 300:, 2:3].values  #后300天的开盘价作为测试集

#归一化
sc = MinMaxScaler(feature_range=(0, 1))  #定义归一化；缩放到（0,1）之间
training_set_scaled = sc.fit_transform(training_set)  #求得训练集的最大最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.fit_transform(test_set)  #利用训练集的属性对测试集进行归一化

x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中的前1975-300=1675天的数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train,第61天的数据作为标签，for循环共构建1975-300-60=1615
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i: 0])
# 打乱训练集顺序
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入的特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]及1615组数据....省略
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
# 测试集：csv表格中后300天数据
# ...省略
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])
# 测试集变arry并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入的特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.3),
    SimpleRNN(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='mean_squared_error')  #损失函数用均方误差
# model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
# 该应用只观测loss数值， 不观测准确率。。。省略

checkpoint_sava_path = "./checkpoint/jijin.ckpt"

if os.path.exists(checkpoint_sava_path + '.index'):
    print('--------load the model--------')
    model.load_weights(checkpoint_sava_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_sava_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()


file = open('./weight.txt', 'w')  #参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
######## predict #######
# 测试集输入模型进行预测
predicted_stock_price = model.predict(x_test)
# 对预测数据还原--从（0-1）反归一化到原始范围
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原--从0-1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set[60:])
# 画出真实数据和预测数据对比曲线
plt.plot(real_stock_price, color='red', label='Jijin Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Jijin Price')
plt.title('Jijin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Jijin Price')
plt.legend()
plt.show()
###### evaluate ######
# 均方误差
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# 均方根误差
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# 平均绝对误差
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差：%.6f' % mse)
print('均方根误差：%.6f' % rmse)
print('平均绝对误差：%.6f' % mae)
