#!/usr/bin/env python
import sys
import os, re
import random
import datetime
import numpy as np
import hickle as hkl
from sklearn import metrics

"""
Bootstrapping_p2p.py

@author: liwenran
"""

###################### Input #######################
if len(sys.argv)<3:
	print '[USAGE] python DataAugmentation_p2e.py cell num_replicate'
	print 'For example, python DataAugmentation_p2e.py demo 3'
	sys.exit()
cell = sys.argv[1]
num_rep = int(sys.argv[2])


################# Initialization ###################
NUM_SEQ = 4
NUM_AUGMENT = 20
NUM_ENSEMBL = 20
RESIZED_PROMOTER_LEN = 1000


############################# Bootstrapping #############################
def bagging():
    os.system('mkdir -p '+cell+'/bagData')
    ## load data: sequence
    enhancer_shape = (-1, 1, RESIZED_PROMOTER_LEN, NUM_SEQ)
    promoter_shape = (-1, 1, RESIZED_PROMOTER_LEN, NUM_SEQ)
    enhancer_D = np.load(cell+'/promoter1_Seq.npz')
    promoter_D = np.load(cell+'/promoter2_Seq.npz')
    Tlabel_D = enhancer_D['label']
    Tenhancer_seq_D = enhancer_D['sequence'].reshape(enhancer_shape).transpose(0, 1, 3, 2)
    Tpromoter_seq_D = promoter_D['sequence'].reshape(promoter_shape).transpose(0, 1, 3, 2)

    ## load data: DNase
    enhancer_shape = (-1, 1, num_rep, RESIZED_PROMOTER_LEN)
    promoter_shape = (-1, 1, num_rep, RESIZED_PROMOTER_LEN)
    enhancer_D = np.load(cell+'/promoter1_DNase.npz')
    promoter_D = np.load(cell+'/promoter2_DNase.npz')
    Tenhancer_expr_D = enhancer_D['expr'].reshape(enhancer_shape)
    Tpromoter_expr_D = promoter_D['expr'].reshape(promoter_shape)

    NUM = Tlabel_D.shape[0]
    for t in range(0, NUM_ENSEMBL):
        print t
        """bootstrap"""
        index = [random.choice(range(NUM)) for i in range(NUM)]
        hkl.dump(index, cell+'/bagData/index_'+str(t)+'.hkl')
        label_D = Tlabel_D[index]
        np.savez(cell+'/bagData/label_'+str(t)+'.npz', label = label_D)
        np.savez(cell+'/bagData/promoter1_Seq_'+str(t)+'.npz', sequence = Tenhancer_seq_D[index], label = label_D)
        np.savez(cell+'/bagData/promoter2_Seq_'+str(t)+'.npz', sequence = Tpromoter_seq_D[index], label = label_D)
        np.savez(cell+'/bagData/promoter1_DNase_'+str(t)+'.npz', expr = Tenhancer_expr_D[index], label = label_D)
        np.savez(cell+'/bagData/promoter2_DNase_'+str(t)+'.npz', expr = Tpromoter_expr_D[index], label = label_D)
        print(label_D.shape[0])
        #
bagging()