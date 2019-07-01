from token import Token

class Sentence(list):

    def __init__(self):
        pass

    def __repr__(self):
        return ' '.join(['%s/%s' % (t.word, t.gold_pos) for t in self])
        #return ' '.join(['%s' % (t.word) for t in self])

    def word_list(self):
        word_list = [t.word for t in self]
        return word_list

    def tag_list(self):
        tag_list = [t.gold_pos for t in self]
        return tag_list

    def add_token(self, line):
        tmp = line.split()
        if len(tmp) > 1:
            word, pos = tmp[0], tmp[1]
        else:
            word, pos = tmp[0], None
        token = Token(self, len(self), word, pos, None)
        self.append(token)



