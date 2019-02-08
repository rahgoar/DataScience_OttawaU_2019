# -*- coding: UTF-8 -*-
from __future__ import print_function
import codecs
import numpy as np
from sklearn.metrics import cohen_kappa_score
import logging
from sklearn import metrics
import pandas as pd
from scipy.stats import fisher_exact 
from scipy.stats import spearmanr

fname='../input.txt'
Hlabels=[]
Rlabels=[]
delimiter=','
for line in codecs.open(fname, 'r', 'UTF-8'):
    row = line.split(delimiter)
    Hlabels.extend(row[1])
    Rlabels.extend(row[2])


with codecs.open( '../text_results.txt', 'w', 'UTF-8') as fo:
        #for Y in X_Y_dic:
        fo.write(str(Rlabels) + '\n')

kappa1 = cohen_kappa_score(Hlabels, Rlabels,weights='linear')
kappa2 = cohen_kappa_score(Hlabels, Rlabels)

corr=spearmanr(Hlabels,Rlabels) # hoping teh wardings go away when I have more data
print(str(corr))
print(kappa1,kappa2)
