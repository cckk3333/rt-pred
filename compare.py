# -*- coding: utf-8 -*-

import sys
from gzip import open
from getopt import getopt
from random import random

from FTRLProximal import *


class ModelRecord(object):

    def __init__(self):
        self.imp = 0
        self.clk = 0
        self.sumlogloss = 0.
    
    def reset(self):
        self.imp = 0
        self.clk = 0
        self.sumlogloss = 0.

    def update(self, p, y):
        self.imp += 1
        self.clk += (y == 1.)
        self.sumlogloss += logloss(p, y)

    def __repr__(self):
        return '%d\t%d\t%f' % (self.imp, self.clk, self.sumlogloss)


class Competer(object):

    def __init__(self, param, price=1.):
        self.param = param
        self.price = price
        self.learner = FtrlProximal(param)
        self.record = ModelRecord()
        self.aggregator = Aggregator(param.aggregation) if param.aggregation else None

    def predict(self,instance):
        x, y = process(instance.copy(), self.param.D, self.aggregator)
        p = self.learner.predict(x)

        # cache
        self.instance = instance
        self.x = x
        self.y = y
        self.p = p
        
        return p

    def update(self, winner=None):
        if not winner or self == winner:
            self.record.update(self.p,self.y)
        
        self.learner.update(self.x,self.p,self.y)
        if self.aggregator:
            self.aggregator.update(instance,self.y)


def select_winner(competerA, competerB):
    pA = competerA.predict(instance)
    pB = competerB.predict(instance)

    bidA = competerA.price * pA
    bidB = competerB.price * pB

    if bidA > bidB or ( bidA == bidB and random() > 1./2 ):
        return competerA
    else:
        return competerB


if __name__ == '__main__':
    optList, argv = getopt(sys.argv[1:], 'hd:A:B:m:')
    if not optList:
        optList.append(('-h',None))

 
    # these two lists store strs for initizing parameters

    datapath = ''
    mode = 0

    optA = []
    optB = []

    for opt, arg in optList:
        if opt == '-h':
            print '''
            \t\t d: datapath. required.
            \t\t A: parameters for model A
            \t\t B: parameters for model B
            \t\t m: compare mode. 0: not share training \t 1: share training . Default: 0
            '''
            exit(0)
        elif opt == '-d':
            datapath = arg
    
        elif opt == '-A':
            optA = getopt(arg.split(),Param.optstr)
            
        elif opt == '-B': 
            optB = getopt(arg.split(),Param.optstr)

        elif opt == '-m':
            mode = int(arg)

    if not datapath:
        print 'data path need to be specified'

    competerA = Competer(Param(optA))
    competerB = Competer(Param(optB))

    print 'ImpA\tClkA\tSumLLA\tImpB\tClkB\tSumLLB'
    
    for date in xrange(20140701,20140731):
        if date == 20140710:
            continue
        for time in xrange(0,1440,10):

            filename = datapath+'/'+str(date)+'/'+str(time)+'.txt.gz'
            
            for instance in datagenerator(filename):    

                winner = select_winner(competerA, competerB) # arguments would cache p, x, y for updating
                
                # not share training data
                if not mode:
                    winner.update()
                # share training data
                else:
                    competerA.update(winner)
                    competerB.update(winner)

            print competerA.record, competerB.record
