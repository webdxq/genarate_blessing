#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import tensorflow as tf  
import os 
import re
import json
import sys
import configuration
import codec
# reload(sys)
# sys.setdefaultencoding('utf8')

def process_data():
	data_config = configuration.DataCofig()
	minlen = data_config.minlen
	maxlen = data_config.maxlen
	cantoneses = open(data_config.cantoneses,'r').readline().split(' ')
	cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
	id2word = data_config.id2word
	word2id = data_config.word2id
	blessings = []  
	all_words = [] 
	cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
	with open(blessing_file, "r") as f:  
		for i,line in enumerate(f):  
		    if i == 0:
		        continue
		    try:  
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
		    except Exception as e:   
		        pass  
		    if i%50000== 0:
		        print(u'处理到%d'%i)

	blessings = sorted(blessings,key=lambda line: len(line))  
	print(u'歌词总行数: %s'% len(blessings))  
	counter = collections.Counter(all_words)  
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  

	print('*******************')
	words, _ = zip(*count_pairs)  
	# 取前多少个常用字  
	print(len(words))

	for i in range(65,66+maxlen-minlen):
	    # print(unicode(chr(i)))
	    words = words[:len(words)] + (unicode(chr(i)),)
	words = words[:len(words)] + (u'[',)  
	words = words[:len(words)] + (u']',)  
	words = words[:len(words)] + (u' ',)
	print(u'词表总数: %s'% len(words))  
	word_num_map = dict(zip(words, range(len(words))))  
	# print(word_num_map[u'['])
	# print(word_num_map[u']'])
	# print(word_num_map[u' '])
	# print(word_num_map[u'A'])
	# print(word_num_map[u'L'])
	to_num = lambda word: word_num_map.get(word, len(words)-1) 
	blessings_vector = [ list(map(to_num,blessing)) for blessing in blessings]  
	# print(blessings_vector[-4:-1])
	# print(blessings_vector[1])
	# for i in blessings[-4:-1]:
	#     print(i)
	# print(blessings[1])
	with open(word2id,'w') as outfile:
	    json.dump(word_num_map,outfile,ensure_ascii=False)
	    # outfile.write('\n')
	with open(id2word,'w') as outfile2:
	    json.dump(words,outfile2,ensure_ascii=False)
	    # outfile2.write('\n') 

	def IsCantonese(line):
	    for i, patten in enumerate(cantonese):
	        if patten.search(line):
	            return True
	    return False

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
	return blessings_vector, word_num_map, word


def LoadDicts():
	data_config = configuration.DataCofig()
	word2id = data_config.word2id
	id2word = data_config.id2word
	with open(word2id,'r') as ToIdf:
	    word_num_map = json.load(ToIdf)
	with open(id2word,'r') as ToWordf:
	    words = json.load(ToWordf)

	return word_num_map,words


# if __name__ == '__main__':
# blessing_file = '/media/pingan_ai/dxq/gen_blessing/dataset/line_lyrics.txt'
# process_data()