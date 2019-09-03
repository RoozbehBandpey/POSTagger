In this work, first a perceptron for classification task was impleneted, with parameter averaging and Passive-Aggressive algorithm during training;

Later the model was turned into a sequential, taking advantage of the sequential dependence of neighboring tags. We used hill climbing method to select the feature
templates automatically, in order to increase the accuracy of the models without increasing drastically the number of features.

These two models were tested in the Part of Speech (POS) tagging and Named Entity Recognition (NER) tasks. In both tasks the sequential model outperformed the perceptron.