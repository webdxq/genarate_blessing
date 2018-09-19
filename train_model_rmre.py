#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import tensorflow as tf  
import os
import sys
import chardet
import re
import json
import time
from datetime import datetime
reload(sys)
sys.setdefaultencoding('utf8')
# os.environ['CUDA_VISIBLE_DEVICES']='1'

#-------------------------------数据预处理---------------------------#  


# poetry_file ='../data/poetry.txt'
minlen = 4
maxlen = 15  
blessing_file ='/home/pingan_ai/dxq/project/gen_blessing/dataset/data/line_lyrics.txt'  
blessings = []  
all_words = [] 
cantoneses = open('/home/pingan_ai/dxq/project/cantonese.txt','r').readline().split(' ')
# print(cantoneses)
cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
LEARNING_RATE_BASE = 0.02
MODEL_SAVE_PATH = '/media/pingan_ai/dxq/gen_blessing/new_model/'
N_GPU = 2
MODEL_NAME = "blessing.ckpt"
EPOCHS = 100
LEARNING_RATE_DECAY = 0.99
filename = blessing_file.split('/')[-1].split('.')[0]
# print(blessing_file)
can_count = 0
MOVING_AVERAGE_DECAY = 0.99
def HasReapeatWord(string):
    flag = False
    for i,char in enumerate(string):
        # print i
        s = i
        m = i+1
        e = i+2 
        # print string[s],string[m],string[e]
        if flag:
            return True
        elif e >= (len(string)-1):
            return False
        else:
            if string[s] == string[m] and string[m] == string[e]:
                # print string[s],string[m],string[e]
                flag = True
            else:
                continue

def IsCantonese(line):
    for i, patten in enumerate(cantonese):
        if patten.search(line)!= None:
            # print(line)
            # can_count = can_count+1
            return True
    return False

with open(blessing_file, "r") as f:  
    for i,line in enumerate(f): 

        if i == 0:
            continue
        # try:  
            # print(line)
        line = line.decode('UTF-8')
        line = line.strip(u'\n')
        line = line.replace(u' ',u'')  
        if u'_' in line or u'(' in line or u'（' in line or u'《' in line or u'[' in line:  
            continue  
        if len(line) < minlen or len(line) > maxlen:  
            continue  
        if IsCantonese(line):
            can_count = can_count+1
            continue
        if HasReapeatWord(line):
            continue         
        all_words += [word for word in line]
        line = u'[' + unicode(chr(len(line)+61)) +line + u']'  
        blessings.append(line)        
        # except Exception as e:   
        #     print('no')  
        if i%100000== 0:
            print(u'处理到%d'%i)
blessings = sorted(blessings,key=lambda line: len(line))  
print(u'歌词总行数: %s'% len(blessings))  
print(can_count)
counter = collections.Counter(all_words)  
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  

print('*******************')
words, _ = zip(*count_pairs)   
print(len(words))
for i in range(65,66+maxlen-minlen):
    words = words[:len(words)] + (unicode(chr(i)),)
words = words[:len(words)] + (u'[',)  
words = words[:len(words)] + (u']',)  
words = words[:len(words)] + (u' ',)
print(u'词表总数: %s'% len(words))  
word_num_map = dict(zip(words, range(len(words))))  
print(word_num_map[u'['])
print(word_num_map[u']'])
print(word_num_map[u' '])
print(word_num_map[u'A'])
print(word_num_map[u'L'])
to_num = lambda word: word_num_map.get(word, len(words)-1) 
blessings_vector = [ list(map(to_num,blessing)) for blessing in blessings]  
to_words = lambda num: words[num]
print(blessings_vector[-4:-1])
print(blessings_vector[1])
for i in blessings[-4:-1]:
    print(i)
print(blessings[1])

with open(filename+'2id_re.json','w') as outfile:
    json.dump(word_num_map,outfile,ensure_ascii=False)
    # outfile.write('\n')
with open(filename+'2word_re.json','w') as outfile2:
    # word2id = dict((value, key) for key,value in word_num_map.iteritems())
    json.dump(words,outfile2,ensure_ascii=False)
    # outfile2.write('\n')

batch_size = 256
n_chunk = len(blessings_vector) // batch_size   
# sys.exit()
class DataSet(object):
    def __init__(self,data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features ,full_batch_labels = self.data_batch(0,batch_size)
            return full_batch_features ,full_batch_labels 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start,end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels

    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(blessings_vector[self._data_index[i]])

        length = max(map(len,batches))
        # print(word_num_map[' '])
        xdata = np.full((end - start,length), word_num_map[']'], np.int32)  
        for row in range(end - start):  
            xdata[row,:len(batches[row])] = batches[row]  
        ydata = np.copy(xdata)  
        ydata[:,:-1] = xdata[:,1:]  
        return xdata,ydata

#---------------------------------------RNN--------------------------------------#  

# 定义RNN  
def neural_network(input_data,model='lstm', rnn_size=128, num_layers=2):  
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == 'lstm':  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  
   
    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)     
    initial_state = cell.zero_state(batch_size, tf.float32)     
    with tf.variable_scope('rnnlm'):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])  
        softmax_b = tf.get_variable("softmax_b", [len(words)])  
        with tf.device("/cpu:0"):  
            embedding = tf.get_variable("embedding", [len(words), rnn_size])  
            inputs = tf.nn.embedding_lookup(embedding, input_data)  
   
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  
   
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)  
    return logits, last_state, probs, cell, initial_state 

def load_model(sess, saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1

def to_word(weights):
    sample = np.argmax(weights)
    return words[sample] 

def train_to_word(x):
    # print(u'x的长度',len(x))
    x_words = map(to_words, x)
    # print(str(x_words).decode("unicode-escape"))
    outstr = ''.join(x_words)
    token = outstr[1]
    outstr = outstr[2:-1]
    print(u'[ '+ token +u' '+ outstr+u' ]')
def AlignSentence(sentence):
    sentence = sentence[:-2]
    sentence_re = ''
    for i in range(len(sentence)):
        if not (sentence[i] >= u'\u4e00' and sentence[i]<=u'\u9fa5'):
            sentence_re += sentence[i]+u' '
        else:
            sentence_re += sentence[i]
    # return u'[ '+sentence[i] + u' ]'
    print sentence_re + u' ]'

def get_loss(input_data, targets, reuse_variables=None):
    # 沿用5.5节中定义的函数来计算神经网络的前向传播结果。
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits, last_state, probs, _, _ = neural_network(input_data) 
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], 
            [targets], 
            [tf.ones_like(targets, dtype=tf.float32)], 
            len(words)
            )
        cost = tf.reduce_mean(loss)  
    return cost

# 计算每一个变量梯度的平均值。
def average_gradients(tower_grads):
    average_grads = []

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来。
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这个将被用于变量的更新。
    return average_grads
# def main(argv=None): 
def main(argv=None): 
    # 将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上。
    TRAINING_STEPS = EPOCHS*n_chunk/N_GPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):   
        input_data = tf.placeholder(tf.int32, [batch_size, None])  
        output_targets = tf.placeholder(tf.int32, [batch_size, None])  
        trainds = DataSet(len(blessings_vector))
        targets = tf.reshape(output_targets, [-1])        
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step, 60000 / batch_size, LEARNING_RATE_DECAY)
        optimizer = tf.train.AdamOptimizer(learning_rate) 
        
        tower_grads = []
        reuse_variables = False
        # 将神经网络的优化过程跑在不同的GPU上。
        
        for i in range(N_GPU):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    
                    cur_loss = get_loss(input_data,targets,reuse_variables)
                    # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
                    # 让不同的GPU更新同一组参数。
                    reuse_variables = True
                    grads = optimizer.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        
        # 计算变量的平均梯度。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() +tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                # 执行神经网络训练操作，并记录训练操作的运行时间。
                start_time = time.time()
                x,y = trainds.next_batch(batch_size)
                _, loss_value = sess.run([train_op, cur_loss],feed_dict={input_data: x, output_targets: y})
                duration = time.time() - start_time
                
                # 每隔一段时间数据当前的训练进度，并统计训练速度。
                if step != 0 and step % 10 == 0:
                    # 计算使用过的训练数据个数。因为在每一次运行训练操作时，每一个GPU
                    # 都会使用一个batch的训练数据，所以总共用到的训练数据个数为
                    # batch大小 × GPU个数。
                    num_examples_per_step = batch_size * N_GPU

                    # num_examples_per_step为本次迭代使用到的训练数据个数，
                    # duration为运行当前训练过程使用的时间，于是平均每秒可以处理的训
                    # 练数据个数为num_examples_per_step / duration。
                    examples_per_sec = num_examples_per_step / duration

                    # duration为运行当前训练过程使用的时间，因为在每一个训练过程中，
                    # 每一个GPU都会使用一个batch的训练数据，所以在单个batch上的训
                    # 练所需要时间为duration / GPU个数。
                    sec_per_batch = duration / N_GPU
    
                    # 输出训练信息。
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    
                    # 通过TensorBoard可视化训练过程。
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)
    
                # 每隔一段时间保存当前的模型。
                if step  == n_chunk:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)
        
main()
# if __name__ == '__main__':
#     tf.app.run()
