import re

class Token:

    def __init__(self, sentence = None, token_id = None, word = None, gold_pos = None, pred_pos = '<NA>'):
        self.sent = sentence
        self.token_id = token_id
        self.word = word
        self.gold_pos = gold_pos
        self.pred_pos = pred_pos

    def __repr__(self):
        return self.word


    def feature_extracter(self, function):
        features = [0]
        features.append(function('WORD:%s' % self.word))

        shapeNC, shapeC = self.shape()

        features.append(function('SHAPENC:%s' % shapeNC))


        features.append(function('SHAPEC:%s' % shapeC))

        features.append(function('PREV+THIS_WORD:%s+%s' % (self.prev_word(1), self.word)))

        features.append(function('NEXT+THIS_WORD:%s+%s' % (self.next_word(1), self.word)))

        return sorted(filter(lambda x: x != None, features))

    def shape(self):
        shapeNC = self.word
        paterns = [['[A-Z]', 'A'], ['[a-z]', 'a'], ['\d', "0"], ['\W', "&"]]

        for i in paterns:
            shapeNC = re.sub(i[0], i[1], shapeNC)

        symbols = ['A', 'a', '&', '0']
        shapeC = shapeNC
        for i in symbols:
            shapeC = re.sub(2 * i + "+", 2 * i, shapeC)
        return shapeNC, shapeC

    def prev_word(self, offset=1):
        if self.token_id - offset < 0:
            return '<BOS>'
        else:
            return self.sent[self.token_id - offset].word

    def next_word(self, offset = 1):
        if self.token_id + offset >= len(self.sent):
            return '<EOS>'
        else:
            return self.sent[self.token_id + offset].word

    def perceptron_HMM_feature(self, token, tag):
        if token == '$':
            #print token
            return 2
        elif token in ['Yes', 'yes', 'No', 'no', 'Ah', 'ah']:
            return 3
        elif token == ',':
            return 4
        elif token == '-':
            return 5
        elif token in ['All', 'such', 'all', 'Such']:
            return 6
        else:
            return 1






