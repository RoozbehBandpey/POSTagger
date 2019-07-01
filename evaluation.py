from terminaltables import AsciiTable


class Evaluation():

    def __int__(self):
        pass

    def Evaluate(self, per_tag, gold_token_list = [], pred_token_list = [], TagSet = []):
        table_data = []
        table_data.append(['Rows',
                           'Tags',
                           'False Positive',
                           'True Positive',
                           'True Negative',
                           'False Negative',
                           'Gold Tag Frq.',
                           'Pred. Tag Frq.',
                           'Precision',
                           'Recall',
                           'Accuracy',
                           'F1-Score'])


        token_list = []
        iNewToken = 0
        while (iNewToken < len(gold_token_list)) :
            #token_list.append(" ".join(token1[0:2]))
            newToken = []
            newToken.append(str(gold_token_list[iNewToken][0]))
            newToken.append(str(gold_token_list[iNewToken][1]))
            newToken.append(str(pred_token_list[iNewToken][1]))
            token_list.append(newToken)
            iNewToken += 1


        allFalseNegative = 0
        allFalsePositive = 0
        allTruePositive = 0
        allTrueNegative = 0
        setCount = 0
        F1Micro = 0.000
        for tag in TagSet:       # 0 ... 48
            #print setCount, tag
            GoldtagFrequency = 0
            PredictedtagFrequency = 0
            tagFalseNegative = 0
            tagFalsePositive = 0
            tagTruePositive = 0
            tagTrueNegative = 0
            corpusCount = 0
            setCount += 1

            for token in token_list:     # 0 ... 26555

                #print token

                if tag == token[1]:

                    GoldtagFrequency += 1
                    if tag == token[2]:
                        tagTruePositive += 1
                        allTruePositive += 1
                        #print "TRUE POSITIVE :  ", tag
                    elif tag != token[2]:
                        tagFalseNegative += 1
                        allFalseNegative += 1

                if tag == token[2]:
                    PredictedtagFrequency += 1
                    if tag != token[1]:
                        tagFalsePositive += 1
                        allFalsePositive += 1

                corpusCount += 1
                tagTrueNegative = len(token_list) - (tagTruePositive + tagFalsePositive + tagFalseNegative)

            #Eval = Evaluation()
            table_data.append([str(setCount),
                               tag,
                               str(int(tagFalsePositive)),
                               str(int(tagTruePositive)),
                               str(int(tagFalseNegative)),
                               str(int(tagTrueNegative)),
                               str(int(GoldtagFrequency)),
                               str(int(PredictedtagFrequency)),
                               str(round(self.Precision(float(tagTruePositive), float(tagFalsePositive)), 3)),
                               str(round(self.Recall(float(tagTruePositive), float(tagFalseNegative)), 3)),
                               str(round(self.Accuracy(float(tagTruePositive), float(tagFalsePositive), float(tagFalseNegative), float(tagTrueNegative)), 3)),
                               str(round(self.F1Score(self.Precision(float(tagTruePositive), float(tagFalsePositive)),
                                                      self.Recall(float(tagTruePositive), float(tagFalseNegative))), 3))])
            F1Micro += round(self.F1Score(self.Precision(float(tagTruePositive), float(tagFalsePositive)),
                                          self.Recall(float(tagTruePositive), float(tagFalseNegative))), 3)

        allTrueNegative = len(token_list) - (allTruePositive + allFalsePositive + allFalseNegative)
        F1total = self.F1Score(self.Precision(float(allTruePositive), float(allFalsePositive)),
                               self.Recall(float(allTruePositive), float(allFalseNegative)))
        #print allTruePositive, allFalsePositive, allTrueNegative
        F1Micro = F1Micro / TagSet.__len__()
        table = AsciiTable(table_data)

        if per_tag == True:
            return table.table
        else:
            return str(round(F1Micro, 3)) , str(round(F1total, 3))


    def tag_true_positive(self, TagListFrequency={}):
        print TagListFrequency
        TagListValues = list(TagListFrequency.values())
        TagListKeys = list(TagListFrequency.keys())
        print TagListValues
        print TagListKeys
        i = 0
        while(i < len(TagListValues)):
            print(str(i + 1)+ "\t" +str(TagListValues[i]) + "\t" + str(TagListKeys[i]))
            i += 1

    def Precision(self, TruePositive, FalsePositive):
        if TruePositive == 0:
            return 0
        else:
            return (TruePositive / (TruePositive + FalsePositive))

    def Recall(self, TruePositive , FalseNegative):
        if TruePositive == 0:
            return 0
        else:
            return (TruePositive / (TruePositive + FalseNegative))

    def Accuracy(self, TruePositive, FalsePositive, FalseNegative,TrueNegative):
        TrueAll = TruePositive + TrueNegative
        FalseAll = FalsePositive + FalseNegative
        return (TrueAll / (TrueAll + FalseAll))

    def F1Score(self, precision , recall):
        if precision == 0:
            return 0
        else:
            return ((2 * precision * recall) / (precision + recall))



