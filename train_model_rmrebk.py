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
os.environ['CUDA_VISIBLE_DEVICES']='1'
reload(sys)
sys.setdefaultencoding('utf8')
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
filename = blessing_file.split('/')[-1].split('.')[0]
# print(blessing_file)
# can_count = 0
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
            continue
        if HasReapeatWord(line):
            continue         
        all_words += [word for word in line]
        line = u'[' + unicode(chr(len(line)+61)) +line + u']'  
        blessings.append(line)        
        # except Exception as e:   
        #     print('no')  
        if i%50000== 0:
            print(u'处理到%d'%i)
blessings = sorted(blessings,key=lambda line: len(line))  
print(u'歌词总行数: %s'% len(blessings))  
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
with open(filename+'2word_re.json','w') as outfile2:
    # word2id = dict((value, key) for key,value in word_num_map.iteritems())
    json.dump(words,outfile2,ensure_ascii=False)

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

def train_neural_network():  
    input_data = tf.placeholder(tf.int32, [batch_size, None])  
    output_targets = tf.placeholder(tf.int32, [batch_size, None])  
    logits, last_state, probs, _, _ = neural_network(input_data)  
    targets = tf.reshape(output_targets, [-1])  
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))  
    # tf.summary.scalar('loss', loss)
    cost = tf.reduce_mean(loss)  
    tf.summary.scalar('cost', cost)
    learning_rate = tf.Variable(0.0, trainable=False)  
    tf.summary.scalar('learning_rate', learning_rate)
    tvars = tf.trainable_variables()  
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)    
    optimizer = tf.train.AdamOptimizer(learning_rate)   
    train_op = optimizer.apply_gradients(zip(grads, tvars))  
    trainds = DataSet(len(blessings_vector))
    config_proto = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
    merged = tf.summary.merge_all()
    with tf.Session(config=config_proto) as sess:
        train_writer = tf.summary.FileWriter('logdir', sess.graph)          
        sess.run(tf.initialize_all_variables())  
        saver = tf.train.Saver(tf.all_variables())
        last_epoch = load_model(sess, saver,'/media/pingan_ai/dxq/gen_blessing/new_model/') 
        step = 0
        sess.run(tf.assign(learning_rate, 0.002) )
        for epoch in range(last_epoch + 1, 200):
            if epoch > 20:
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** (epoch))) )  
            #sess.run(tf.assign(learning_rate, 0.01))          
            all_loss = 0.0 
            for batche in range(n_chunk): 
                step+=1
                x,y = trainds.next_batch(batch_size)
                summary,train_loss, _ , _ ,probs_= sess.run([merged, cost, last_state, train_op, probs], feed_dict={input_data: x, output_targets: y})  
                
                all_loss = all_loss + train_loss 
                train_writer.add_summary(summary,step)
                if batche % 500 == 1:
                    #print(epoch, batche, 0.01,train_loss) 
                    print(epoch, batche, 0.002 * (0.97 ** (epoch-20)),train_loss) 

            saver.save(sess, '/media/pingan_ai/dxq/gen_blessing/new_model/', global_step=epoch) 
            print (epoch,' Loss: ', all_loss * 1.0 / n_chunk) 
            # print(type(probs_))
            probs_= probs_.reshape(batch_size, -1, len(words))
            y_words = [ list(map(to_word, prob_)) for prob_ in probs_] 
            train_to_word(x[-1])
            AlignSentence(''.join(y_words[-1]))
            print('********************')
            train_to_word(x[-2])
            AlignSentence(''.join(y_words[-2]))
            del probs_
            del y_words

train_neural_network()