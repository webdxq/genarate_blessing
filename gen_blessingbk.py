#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import tensorflow as tf  
import os 
import re
import json
import sys
import beam_search
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['CUDA_VISIBLE_DEVICES']='0'

minlen = 4
maxlen = 15  
blessing_file ='/home/pingan_ai/dxq/project/gen_blessing/dataset/data/line_lyrics.txt'  

cantoneses = open('/home/pingan_ai/dxq/project/cantonese.txt','r').readline().split(' ')
cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
char2num = {4:'A',5:'B',6:'C',7:'D',8:'E',9:'F',10:'G',11:'H',12:'I',13:'J',14:'K',15:'L'}
id2word = '/home/pingan_ai/dxq/project/LstmBlessing/codebk/line_lyrics_2word.json'
# id2word = './line_lyrics_2id.json'
# word2id = None
word2id = '/home/pingan_ai/dxq/project/LstmBlessing/codebk/line_lyrics_2id.json'
# def IsCantonese(line):
#     for i, patten in enumerate(cantonese):
#         if patten.search(line):
#             return True
#     return False

# if word2id == None:
#     with open(blessing_file, "r") as f:  
#         for i,line in enumerate(f):  
#             if i == 0:
#                 continue
#             try:  
#                 line = line.decode('UTF-8')
#                 line = line.strip(u'\n')
#                 line = line.replace(u' ',u'')  
#                 if u'_' in line or u'(' in line or u'（' in line or u'《' in line or u'[' in line:  
#                     continue  
#                 if len(line) < minlen or len(line) > maxlen:  
#                     continue  
#                 if IsCantonese(line):
#                     continue
#                 if HasReapeatWord(line):
#                     continue
#                 all_words += [word for word in line]
#                 line = u'[' + unicode(chr(len(line)+61)) +line + u']'  
#                 blessings.append(line)        
#             except Exception as e:   
#                 pass  
#             if i%50000== 0:
#                 print(u'处理到%d'%i)

#     blessings = sorted(blessings,key=lambda line: len(line))  
#     print(u'歌词总行数: %s'% len(blessings))  
#     counter = collections.Counter(all_words)  
#     count_pairs = sorted(counter.items(), key=lambda x: -x[1])  

#     print('*******************')
#     words, _ = zip(*count_pairs)  
#     # 取前多少个常用字  
#     print(len(words))

#     for i in range(65,66+maxlen-minlen):
#         # print(unicode(chr(i)))
#         words = words[:len(words)] + (unicode(chr(i)),)
#     words = words[:len(words)] + (u'[',)  
#     words = words[:len(words)] + (u']',)  
#     words = words[:len(words)] + (u' ',)
#     print(u'词表总数: %s'% len(words))  
#     word_num_map = dict(zip(words, range(len(words))))  
#     # print(word_num_map[u'['])
#     # print(word_num_map[u']'])
#     # print(word_num_map[u' '])
#     # print(word_num_map[u'A'])
#     # print(word_num_map[u'L'])
#     to_num = lambda word: word_num_map.get(word, len(words)-1) 
#     blessings_vector = [ list(map(to_num,blessing)) for blessing in blessings]  
#     to_words = lambda num: words[num]
#     # print(blessings_vector[-4:-1])
#     # print(blessings_vector[1])
#     # for i in blessings[-4:-1]:
#     #     print(i)
#     # print(blessings[1])
#     with open(filename+'2id.json','a') as outfile:
#         json.dump(word_num_map,outfile,ensure_ascii=False)
#         outfile.write('\n')
#     with open(filename+'2word.json','a') as outfile2:
#         json.dump(words,outfile2,ensure_ascii=False)
#         outfile2.write('\n') 
# else:
#     with open(word2id,'r') as ToIdf:
#         word_num_map = json.load(ToIdf)
#     with open(id2word,'r') as ToWordf:
#         words = json.load(ToWordf)
#     to_words = lambda num: words[num]


batch_size = 1

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
                flag = True
            else:
                continue
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
    with tf.variable_scope('rnnlm',reuse=tf.AUTO_REUSE):  
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

def to_word(weights):  
    t = np.cumsum(weights)  
    s = np.sum(weights)  
    nprand = np.random.rand(1)*s
    sample = int(np.searchsorted(t, nprand))  
    try:
        # print (words[sample] )
        # print('back',word_num_map(words[sample]))
        return words[sample] 
    except IndexError as e:
        with open('IndexError.txt','a') as wf:          
            wf.write('weights is:\n ')
            wf.write(str(weights))
            wf.write('index is : %d \n'%sample)
            wf.write('nprand is : %f \n'%nprand)
        # print sample
        # sys.exit()
# def to_word(weights):
#     print(len(weights))
#     sample = np.argmax(weights)
#     print(sample)
#     return words[sample]   

def train_to_word(x,words):
    # print(u'x的长度',len(x))
    to_words = lambda num: words[num]
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
    print u'[ '+ sentence_re + u' ]'

def gen_blessing(heads, length,num_sentences):  
    # def to_word(weights):  
    #     t = np.cumsum(weights)  
    #     s = np.sum(weights)  
    #     sample = int(np.searchsorted(t, np.random.rand(1)*s))  
    #     return words[sample]  
    with tf.Graph().as_default():
        input_data = tf.placeholder(tf.int32, [batch_size, None])  
        output_targets = tf.placeholder(tf.int32, [batch_size, None])  
        _, last_state, probs, cell, initial_state = neural_network(input_data)
        config_proto = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        saver = tf.train.Saver(tf.global_variables()) 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=config_proto) as sess: 
            sess.run(init_op)  
            checkpoint = tf.train.latest_checkpoint('/media/pingan_ai/dxq/gen_blessing/models_1/')
            print(checkpoint)
            saver.restore(sess, checkpoint)
            poem = ''
            for i in range(int(num_sentences)):
                flag = True
                while flag:
                    state_ = sess.run(cell.zero_state(1, tf.float32)) 
                    sentence=heads
                    inputs_start = [u'[',char2num[int(length)]]
                    for head in heads:
                        inputs_start.append(head)
                    x = np.array([list(map(word_num_map.get, inputs_start))])  
                    [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                    word = to_word(probs_[-1])
                    sentence += word  
                    while word != ']' and word != ' ':  
                        x = np.zeros((1,1))  
                        x[0,0] = word_num_map[word]  
                        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                        word = to_word(probs_)
                        sentence += word 
                        # print(word)
                    sentence += u'\n'
                    poem += sentence
                    flag = False
            print(poem)
        return

def beamsearch_blessing():
    with tf.Graph().as_default():
        input_data = tf.placeholder(tf.int32, [batch_size, None])  
        output_targets = tf.placeholder(tf.int32, [batch_size, None])  
        _, last_state, probs, cell, initial_state = neural_network(input_data)
        config_proto = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        saver = tf.train.Saver(tf.global_variables()) 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=config_proto) as sess: 
            sess.run(init_op)  
            checkpoint = tf.train.latest_checkpoint('/media/pingan_ai/dxq/gen_blessing/models_1/')

            saver.restore(sess, checkpoint)
            state_ = sess.run(cell.zero_state(1, tf.float32)) 
            sentence=heads
            inputs_start = [u'[',char2num[int(length)]]
            for head in heads:
                inputs_start.append(head)
            x = np.array([list(map(word_num_map.get, inputs_start))]) 
            captions = generator.beam_search(sess, x, state_)
            poem = ''
            for i in range(int(num_sentences)):
                flag = True
                while flag:
                    
                    
                     
                    [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                    word = to_word(probs_[-1])
                    sentence += word  
                    while word != ']' and word != ' ':  
                        x = np.zeros((1,1))  
                        x[0,0] = word_num_map[word]  
                        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                        word = to_word(probs_)
                        sentence += word 
                        # print(word)
                    sentence += u'\n'
                    poem += sentence
                    flag = False
            print(poem)
        return

if __name__ == '__main__':
      
    # print(gen_head_poetry(u'为为为为',5)) 
    # 
    sentence_head = raw_input('输入分句首字：')
    # print(chardet.detect(sentence_head))
    while sentence_head:
        # pass
        # print(sentence_head)
        length = raw_input('输入分句长度：')
        num_sentences = raw_input('输入分句数量：')
        # tf.reset_default_graph()
        gen_blessing(sentence_head.decode('utf-8'),length,num_sentences)
        sentence_head = raw_input('输入分句首字：')
    
