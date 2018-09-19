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
import configuration
import model
from data_utils import LoadDicts
from tensorflow.python.tools import inspect_checkpoint as chkp
import math
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['CUDA_VISIBLE_DEVICES']='0'


char2num = {4:'A',5:'B',6:'C',7:'D',8:'E',9:'F',10:'G',11:'H',12:'I',13:'J',14:'K',15:'L'}
test_config = configuration.TestingConfig()
batch_size = test_config.bach_size
word_num_map, words = LoadDicts()

# def to_word(weights, words):  
#     t = np.cumsum(weights)  
#     s = np.sum(weights)  
#     nprand = np.random.rand(1)*s
#     sample = int(np.searchsorted(t, nprand))  
#     try:
#         # print (words[sample] )
#         # print('back',word_num_map(words[sample]))
#         return words[sample] 
#     except IndexError as e:
#         with open('IndexError.txt','a') as wf:          
#             wf.write('weights is:\n ')
#             wf.write(str(weights))
#             wf.write('index is : %d \n'%sample)
#             wf.write('nprand is : %f \n'%nprand)

def to_word(weights,words):
    sample = np.argmax(weights)
    return words[sample] 

        # print sample
        # sys.exit()
# def to_word(weights):
#     print(len(weights))
#     sample = np.argmax(weights)
#     print(sample)
#     return words[sample]   

def train_to_word(x, words):
    # print(u'x的长度',len(x))
    to_words = lambda num: words[num]
    x_words = map(to_words, x)
    # print(str(x_words).decode("unicode-escape"))
    outstr = ''.join(x_words)
    token = outstr[1]
    outstr = outstr[2:-1]
    return u'[ '+ token +u' '+ outstr+u' ]'

def AlignSentence(sentence):
    sentence = sentence[:-2]
    sentence_re = ''
    for i in range(len(sentence)):
        if not (sentence[i] >= u'\u4e00' and sentence[i]<=u'\u9fa5'):
            sentence_re += sentence[i]+u' '
        else:
            sentence_re += sentence[i]
    # return u'[ '+sentence[i] + u' ]'
    return u'[ '+ sentence_re + u' ]'

def gen_blessing(heads, length,num_sentences):  
    with tf.Graph().as_default():
        # chkp.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(test_config.checkpoint_path), tensor_name='', all_tensors=True)
        input_data = tf.placeholder(tf.int32, [batch_size, None])  
        # print('input_data')
        output_targets = tf.placeholder(tf.int32, [batch_size, None])  
        _, last_state, probs, cell, initial_state= model.neural_network(input_data,len(words),batch_size)

        config_proto = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        saver = tf.train.Saver(tf.global_variables()) 
        checkpoint = tf.train.latest_checkpoint(test_config.checkpoint_path)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=config_proto) as sess: 
            sess.run(init_op)  
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
                    # for w in probs_:
                    #     print(to_word(w, words))
                    word = to_word(probs_[-1], words)
                    # print(word)
                    sentence += word  
                    while word != ']' and word != ' ':  
                        x = np.zeros((1,1))  
                        x[0,0] = word_num_map[word]  
                        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                        word = to_word(probs_, words)
                        sentence += word 
                        # print(word)
                    sentence += u'\n'
                    poem += sentence
                    flag = False
            print(poem)
    return

def beamsearch_blessing(heads, length,num_sentences):
    with tf.Graph().as_default():
        input_data = tf.placeholder(tf.int32, [batch_size, None])  
        output_targets = tf.placeholder(tf.int32, [batch_size, None])  
        _, last_state, probs, cell, initial_state = model.neural_network(input_data,len(words),batch_size)
        config_proto = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        saver = tf.train.Saver(tf.global_variables()) 
        checkpoint = tf.train.latest_checkpoint(test_config.checkpoint_path)
        generator = beam_search.CaptionGenerator(word_num_map[u']'],int(num_sentences))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=config_proto) as sess: 
            sess.run(init_op)  
            saver.restore(sess, checkpoint)
            state_ = sess.run(cell.zero_state(1, tf.float32)) 
            sentence=heads
            inputs_start = [u'[',char2num[int(length)]]
            for head in heads:
                inputs_start.append(head)
            x = np.array([list(map(word_num_map.get, inputs_start))]) 
            captions = generator.beam_search(sess, probs, last_state, x, state_,initial_state,input_data)
            num = 1
            for i, caption in enumerate(captions):
            # Ignore begin and end words.
                # print(caption.sentence)
                sentence = [words[w] for w in caption.sentence[1:-1]]
                sentence = "".join(sentence)
                # sentence = [(vocab.id_to_word(w) +' ') for w in caption.sentence[1:-1]]
                # sentence = "".join(sentence)
                # print(caption.logprob)
                # print(i,sentence)
                # if i == 0:
                print("%d) %s (p=%f)" % (num,sentence, math.exp(caption.logprob)))
                print('\n')
                num = num + 1
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
        beamsearch_blessing(sentence_head.decode('utf-8'),length,num_sentences)
        sentence_head = raw_input('输入分句首字：')
    
