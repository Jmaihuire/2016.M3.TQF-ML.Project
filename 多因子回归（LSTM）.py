# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:28:19 2017

@author: zhoukw
"""

import os 
import pandas as pd 
import numpy as np


# 提取所有单因子数据
filepath = 'd:\\Users\\zhoukw\\Desktop\\暑期PROJECT\\data\\单因子测试\\单因子数据\\'
[files] = os.walk(filepath)
allfiles = files[2]

for i,file in enumerate(allfiles):
    if i==0:
       dataall = pd.read_excel(filepath+allfiles[i])
    if i == 1:
       dataall2 = pd.read_excel(filepath+allfiles[i])
       data = pd.merge(dataall,dataall2,on=['date','industry_name','industry_code'])
    if i>1:
       dataall2 = pd.read_excel(filepath+allfiles[i])
       data = pd.merge(data,dataall2,on=['date','industry_name','industry_code'])

data.fillna(method='backfill',inplace=True) # 处理因子缺失值  
 
# 读取return 数据
filepath2 = 'd:\\Users\\zhoukw\\Desktop\\暑期PROJECT\\data\\'
returndata = pd.read_excel(filepath2 + 'industry_data_pct_chg_M.xlsx')

mergedata = pd.merge(returndata,data,on=['date','industry_name','industry_code'])

data = mergedata.iloc[:,3:].values

# 深度学习 tensorflow 框架
import tensorflow as tf

#定义常量
time_step = 20      #时间步
batch_size = 60     #每一批次训练多少个样例
rnn_unit = 10       #hidden layer units 隐藏层
input_size = data.shape[1]-1      
output_size = 1
lr = 0.0006         #学习率


#——————————获取训练集——————————
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=3000):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,1:]
       y=normalized_train_data[i:i+time_step,0,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#——————————获取测试集——————————
def get_test_data(time_step=20,test_begin=3000):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,1:]
       y=normalized_test_data[i*time_step:(i+1)*time_step,0]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,1:]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,0]).tolist())
    return mean,std,test_x,test_y

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }


def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    with tf.variable_scope("lstm"):
         cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0)
         init_state=cell.zero_state(batch_size,dtype=tf.float32)
         output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
         output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#——————————————————训练模型——————————————————
def train_lstm(batch_size=60,time_step=20,train_begin=0,train_end=3000):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(100):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_,)
            if i % 200==0:
                modelpath = 'd:\\Users\\zhoukw\\Desktop\\LSTMmodel\\' #模型保存的位置
                print("保存模型：",saver.save(sess,modelpath,global_step=i))

train_lstm()

#————————————————预测模型————————————————————

def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    with tf.variable_scope("lstm",reuse=True):
         cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0)
         init_state=cell.zero_state(batch_size,dtype=tf.float32)
         output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
         output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

modelpath = 'd:\\Users\\zhoukw\\Desktop\\LSTMmodel\\' #模型保存的位置
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(checkpoint_dir= modelpath)
        saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[0]+mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]
    return test_y,test_predict
test_y,test_predict = prediction()
acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差

import matplotlib.pyplot as plt
plt.plot(test_y)
plt.plot(test_predict)