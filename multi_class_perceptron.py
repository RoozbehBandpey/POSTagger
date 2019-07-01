from __future__ import division
import numpy as numpy
from corpus import Corpus
from evaluation import *
import cPickle
import gzip
import operator
from token import Token
import time
import itertools



class multi_class_perceptron:

    def __init__(self, modelfile = None):#Initializing the multi_class_perceptron
        self.table_statistics = []
        self.table_statistics.append(['Features', 'Tags', 'Words', 'Sentences'])
        if modelfile:
            self.load(modelfile)
        else:
            self.feature_dictionary = {'#': 0}
            self.pos_dictionary = {}
            self.pos_dictionary_reverse = {}


    def feature_constructor(self, feature):
        if feature not in self.feature_dictionary:
            self.feature_dictionary[feature] = len(self.feature_dictionary)
        return self.feature_dictionary[feature]

    def pos_constructor(self, pos):
        if pos not in self.pos_dictionary:
            i = len(self.pos_dictionary)
            self.pos_dictionary[pos] = i
            self.pos_dictionary_reverse[i] = pos
        return self.pos_dictionary[pos]

    def return_features(self, feature):
        return self.feature_dictionary.get(feature, None)

    def weights_constructor(self):
        self.feature_length = len(self.feature_dictionary)
        self.pos_length = len(self.pos_dictionary)

        self.weights = numpy.full((len(self.feature_dictionary), len(self.pos_dictionary)), 5.00)

        return self.pos_length , self.feature_length


    def weight_scores(self, features):
        return numpy.sum(self.weights[i] for i in features)

    def predict(self, scores):
        guessed_score = None
        min_tag = 0
        for i, pos_tag in enumerate(scores):
            #print i
            #print p
            if pos_tag > min_tag:
                guessed_score = i
                min_tag = pos_tag
        return guessed_score

    def update(self, feature, pos_tag, predicted_tag):
        for i in feature:
            #self.weights[i, pos_tag] *= 1.1
            #self.weights[i, predicted_tag] *= 0.9
            self.weights[i, pos_tag] += 1
            self.weights[i, predicted_tag] -= 1


    def save(self, modelfile):
        temp_file = gzip.open(modelfile, 'wb')
        cPickle.dump(self.weights, temp_file, -1)
        cPickle.dump(self.feature_dictionary, temp_file, -1)
        cPickle.dump(self.pos_dictionary, temp_file, -1)
        cPickle.dump(self.pos_dictionary_reverse, temp_file, -1)
        temp_file.close()

    def load(self, modelfile):
        temp_file = gzip.open(modelfile, 'rb')
        self.weights = cPickle.load(temp_file)
        self.feature_dictionary = cPickle.load(temp_file)
        self.pos_dictionary = cPickle.load(temp_file)
        self.pos_dictionary_reverse = cPickle.load(temp_file)
        temp_file.close()

    def return_pos_reverse(self, index):
        return self.pos_dictionary_reverse[index]

    def train(self, train_file, iteration):
        print("________________________________________________________________________________________________tarin starts")
        model = multi_class_perceptron()
        c = Corpus()

        instances = []
        sentence_count = 0
        for sentence in c.read_sentence(train_file):
            sentence_count += 1
            for token in sentence:
                feature = token.feature_extracter(model.feature_constructor)
                # print feature
                instances.append((feature, model.pos_constructor(token.gold_pos)))

        weights_statistics = model.weights_constructor()

        self.table_statistics.append([str(weights_statistics[1]),
                                      str(weights_statistics[0]),
                                      str(len(instances)),
                                      str(sentence_count)])

        table = AsciiTable(self.table_statistics)
        print(table.table)


        for iter_round in range(iteration):
            start = time.time()
            for (feature, pos_tag) in instances:
                #print (feature, pos_tag)
                score = model.weight_scores(feature)
                #print score
                predicted_tag = model.predict(score)
                #print predicted_tag
                if predicted_tag != pos_tag:
                    model.update(feature, pos_tag, predicted_tag)
            end = time.time()
            print 'Iteration'+'\t'+str(iter_round+1)+'\t'+'done.', " runs at:", end - start, "seconds"
            model_file = 'Output_Files\\model_'+str(iter_round+1)+'.dump'
            model.save(model_file)
        print("________________________________________________________________________________________________tarin ends")


    def tagger(self, filename, iteration):
        print "________________________________________________________________________________________________Perceptron tagger starts"
        for iter_round in range(iteration):
            model_file = 'Output_Files\\model_' + str(iter_round + 1) + '.dump'
            print 'Reading from file'+'\t'+model_file.split('\\')[1]
            model = multi_class_perceptron(model_file)
            c = Corpus()

            output = open('Output_Files\\dev-predicted.col', 'w')
            for sentence in c.read_sentence(filename):
                for token in sentence:
                    feature = token.feature_extracter(model.return_features)
                    score = model.weight_scores(feature)
                    predicted_tag = model.predict(score)
                    pos_tag = model.pos_constructor(token.gold_pos)
                    output.write('%s\t%s\n' % (token.word, model.return_pos_reverse(predicted_tag)))

                output.write('\n')
            output.close()

            Cgold = Corpus("Input_Files\\test.col")
            GoldWordTagList = Cgold.Tokenize(Cgold)

            Cpred = Corpus("Output_Files\\dev-predicted.col")
            PredWordTagList = Cpred.Tokenize(Cpred)

            Ctag = Corpus("Input_Files\\test.col")
            TagSet = Ctag.tagSet(Ctag)

            eval = Evaluation()
            per_tag = False
            f_measure = eval.Evaluate(per_tag, GoldWordTagList, PredWordTagList, TagSet)

            print 'F-Measure Micro:'+'\t'+f_measure[0]
            print 'F-Measure Macro:'+'\t'+f_measure[1]
            print
        final_eval = Evaluation()
        f_per_tag = True
        per_tag_table = final_eval.Evaluate(f_per_tag, GoldWordTagList, PredWordTagList, TagSet)
        print per_tag_table

        print "________________________________________________________________________________________________Perceptron tagger ends"


    def viterbi_tagger(self, test_file):
        print "________________________________________________________________________________________________viterbi_tagger starts"
        c_3 = Corpus("Input_Files\\train.col")

        stream_emission_matrix = gzip.open("Output_Files\\emission_matrix.dump", 'rb')
        emission_matrix = cPickle.load(stream_emission_matrix)
        stream_emission_matrix.close()




        stream_transition_matrix = gzip.open("Output_Files\\transition_matrix.dump", 'rb')
        transition_matrix = cPickle.load(stream_transition_matrix)
        stream_transition_matrix.close()

        for x in transition_matrix:
            for p in transition_matrix[x]:
                if transition_matrix[x][p] > 0.2:
                    print p, x, transition_matrix[x][p]

        sentence_count = 0
        word_count = 0

        output = open('Output_Files\\dev-predicted-viterbi.col', 'w')
        for sentence in c_3.read_sentence(test_file):
            observation = sentence.word_list()
            sentence_count += 1
            word_count += len(observation)
            #print observation
            states = sentence.tag_list()
            #print states
            #for word in observation:
             #   if word in emission_matrix:
            #        states = states + emission_matrix[word].keys()
            #    else:
            #        states = states + ['NN']

            states = list(set(states))
            #states.insert(0, '<S>')

            #start = time.time()
            prediction = self.viterbi_smoothing(observation, states, emission_matrix, transition_matrix)
            #end = time.time()
            #print 'Sentence '+str(sentence_count)+' at', end - start

            for i in range(len(prediction[0])):
                output.write('%s\t%s\n' % (prediction[1][i], prediction[0][i]))
            output.write('\n')

        output.close()

        Ctag = Corpus("Input_Files\\test.col")
        TagSet = Ctag.tagSet(Ctag)


        self.table_statistics.append([str(emission_matrix.__len__()),
                                      str(len(TagSet)),
                                      str(word_count),
                                      str(sentence_count)])
        table = AsciiTable(self.table_statistics)
        print(table.table)
        print "________________________________________________________________________________________________viterbi_tagger ends"

        Cgold = Corpus("Input_Files\\test.col")
        GoldWordTagList = Cgold.Tokenize(Cgold)

        Cpred = Corpus("Output_Files\\dev-predicted-viterbi.col")
        PredWordTagList = Cpred.Tokenize(Cpred)



        eval = Evaluation()
        per_tag = False
        f_measure = eval.Evaluate(per_tag, GoldWordTagList, PredWordTagList, TagSet)

        print 'F-Measure Micro:' + '\t' + f_measure[0]
        print 'F-Measure Macro:' + '\t' + f_measure[1]
        print


        final_eval = Evaluation()
        f_per_tag = True
        per_tag_table = final_eval.Evaluate(f_per_tag, GoldWordTagList, PredWordTagList, TagSet)
        print per_tag_table


    def viterbi_smoothing(self, observation, states, emission_matrix, transition_matrix):
        #, observation, states, emission_matrix, transition_matrix):
        # initialization
        #print observation
        #print states
        #observation = ['Janet', 'will', 'back', 'the', 'billlklklk', '.']
        #states = ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB' , 'DT']
        ##states = ['NNP', 'NNS', 'VBZ', '.']

        #observation = ['To', 'be', 'sure', ',', 'Kao', 'would', "n't", 'have', 'an', 'easy', 'time', 'taking', 'U.S.', 'market',
        # 'share', 'away', 'from', 'the', 'mighty', 'P&G', ',', 'which', 'has', 'about', '23', '%', 'of', 'the',
        # 'market', '.']
        #states = ['MD', 'VB', 'VBG', 'JJ', 'NN', 'RBR', ',', '.', 'VBN', 'TO', 'VBP', 'WDT', 'RB', 'IN', 'RP', 'DT', 'CD', 'NNS',
         #'VBZ', 'NNP']

        #observation =['In', 'addition', 'to', '$', '33', 'million', 'compensatory', 'damages', ',', 'the', 'suit', 'seeks', '$',
        # '100', 'million', 'in', 'punitive', 'damages', '.']
        #states = ['VB', 'RP', '$', 'NN', 'FW', 'RBR', 'DT', 'CD', 'TO', 'RB', 'IN', 'VBZ', 'JJ', '.', 'NNS', ',', 'NNP']
#
        #c_1 = Corpus("Input_Files\\train.col")
        #c_2 = Corpus("Input_Files\\train.col")
#
        #emission_matrix = c_1.dictionary_maker(c_1)
        #transition_matrix = c_2.bigram_preprocess(c_2)

        #print transition_matrix.keys()

        #for x in observation:
        #    print x, emission_matrix[x]

        #for wordem in observation:
        #    print wordem, emission_matrix[wordem]

        #for tagtm in states:
        #    print tagtm, transition_matrix[tagtm]

        viterbi = numpy.zeros((len(observation), len(states)))

        for ini_state in states:
            if observation[0] in emission_matrix:
                if ini_state in emission_matrix[observation[0]].keys():
                    viterbi[0][states.index(ini_state)] = \
                        emission_matrix[observation[0]][ini_state] * transition_matrix[ini_state]['<S>']
            else:
                viterbi[0][states.index(ini_state)] = transition_matrix[ini_state]['<S>']

        #print viterbi

        #a = [1,2,3,4]
        #b = [2,2,2,2]
        #c = numpy.multiply(a,b)
        #print c

        # H@des was here
        # He turned coffee to viterbi HMM decoder

        #start = time.time()

        token = Token()

        for word in observation:
            if word in emission_matrix:
                if observation.index(word) == 0:
                    pass
                else:
                    # print word
                    for prev_state in states:
                        for nex_state in states:
                            feature_factor = token.perceptron_HMM_feature(word, nex_state)
                            if nex_state in emission_matrix[word].keys():
                                #print word, prev_state, nex_state, max(viterbi[observation.index(word) - 1]), transition_matrix[nex_state][prev_state] , emission_matrix[word][nex_state]
                                viterbi[observation.index(word)][states.index(nex_state)] = (transition_matrix[prev_state][nex_state] * \
                                                                                            emission_matrix[word][nex_state]) + \
                                                                                            feature_factor

                            else:
                                viterbi[observation.index(word)][states.index(prev_state)] = 0.0
            else:
                if observation.index(word) == 0:
                    pass
                else:
                    # print word
                    for prev_state in states:
                        for nex_state in states:
                            feature_factor = token.perceptron_HMM_feature(word, nex_state)
                            viterbi[observation.index(word)][states.index(nex_state)] =  transition_matrix[prev_state][nex_state] + \
                                                                                         feature_factor

        most_probable_tag_sequence = []
        for word in observation:
            if word != ".":
                possible_tags = []
                for state in states:
                    possible_tags.append(viterbi[observation.index(word)][states.index(state)])
                index, value = max(enumerate(possible_tags), key=operator.itemgetter(1))
                most_probable_tag_sequence.append(states[index])
            else:
                most_probable_tag_sequence.append(".")

        #print viterbi
        #print most_probable_tag_sequence

        #print most_probable_tag_sequence
        #print observation
        return most_probable_tag_sequence, observation