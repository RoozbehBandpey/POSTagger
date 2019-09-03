from __future__ import division
from sentence import Sentence
from collections import Counter
import numpy as numpy
import cPickle
import gzip


class DataHelper():
	
	def __init__(self, filePath=None):
		"""
		Data helper responsible for handling data for preprocessing for the learning algorithms
		:param filePath:
		"""
		if filePath:
			self.file = open(filePath, 'r')
		else:
			pass
	
	
	def read_sentence(self, filename):
		# print("________________________________________________________________________________________________read_sentence starts")
		sentence = Sentence()
		for line in open(filename):
			line = line.strip()
			if line:
				sentence.add_token(line)
			elif len(sentence) != 1:
				yield sentence
				sentence = Sentence()  # print("________________________________________________________________________________________________read_sentence ends")
	
	
	def word_spliter(self, word_POS_pair):
		return (str(word_POS_pair).split("\t"))[0]
	
	
	def POS_spliter(self, word_POS_pair):
		return ((str(word_POS_pair).split("\t"))[1]).strip('\n')
	
	
	def Tokenize(self, File):
		# print "_________________________________________________________________________________Tokenize STARTS"#
		token_list = File.file.readlines()
		word_tag_list = []
		for token in token_list:
			if token != "\n":
				pair = []
				pair.append(self.word_spliter(token))
				pair.append(self.POS_spliter(token))
				if '' not in pair:
					word_tag_list.append(pair)
		
		# print "_________________________________________________________________________________Tokenize ENDS"
		return word_tag_list
	
	
	def tagSet(self, File):
		# print "_________________________________________________________________________________tagSet STARTS"
		token_list = File.file.readlines()
		tag_set = []
		for token in token_list:
			if token != "\n":
				tag = self.POS_spliter(token)
				if tag != '':
					tag_set.append(tag)
		
		# print "_________________________________________________________________________________tagSet ENDS"
		return set(tag_set)
	
	
	def dictionary_maker(self, File):
		word_tag_list = self.Tokenize(File)
		
		tags = []
		for word_tag in word_tag_list:
			tags.append(word_tag[1])
		
		tags_frequency = dict(Counter(tags))
		
		temp_file = gzip.open("dataset\\dict.dump", 'rb')
		word_dict = cPickle.load(temp_file)
		temp_file.close()
		
		for item in word_dict:
			for tag in tags_frequency:
				if tag in word_dict[item]:
					word_dict[item][tag] = word_dict[item][tag] / tags_frequency[
						tag]  # print item_2, word_dict[item][item_2]
		
		return word_dict
	
	
	def bigram_preprocess(self, File):
		word_tag_list = self.Tokenize(File)
		
		tags = ['<S>']
		for word_tag in word_tag_list:
			tags.append(word_tag[1])
			if word_tag[1] == '.':
				tags.append('<S>')
		
		tags.pop()
		
		tags_frequency = dict(Counter(tags))
		
		#
		tag_bigram = zip(tags[1:], tags)
		
		#
		tag_bigram_frequency = dict(Counter(tag_bigram))
		#
		tag_bigram_dict = {}
		for tag_1 in tags_frequency:
			tmp_dict = {}
			for tag_2 in tags_frequency:
				if (tag_1, tag_2) in tag_bigram_frequency:
					tmp_dict.__setitem__(tag_2, tag_bigram_frequency[(tag_1, tag_2)] / tags_frequency[tag_2])
				else:
					tmp_dict.__setitem__(tag_2, 0.0)
			tag_bigram_dict.__setitem__(tag_1, tmp_dict)
		#
		
		return tag_bigram_dict  #