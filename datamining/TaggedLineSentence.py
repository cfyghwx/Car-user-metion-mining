import os
import random
from gensim.models.doc2vec import TaggedDocument
import pandas as pd


#处理文本，将文本分词并转化并标注
class TaggedLineSentence(object):

    def to_array(self,list):
        self.sentences=[]
        i=0
        for sen in list:
            self.sentences.append(TaggedDocument(sen,[i]))
            i+=1
        # print(i)
        return self.sentences

    def perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled