import sys
from gzip import open
from getopt import getopt
from FTRLProximal import *
from random import random

class ModelRecord(object):

    def __init__(self):
        self.imp = 0
        self.clk = 0
        self.sumLL = 0.
    
    def reset(self):
        self.imp = 0
        self.clk = 0
        self.sumLL = 0.

    def update(self, p, y):
        self.imp += 1
        self.clk += y
        self.sumLL = logloss(p, y)


if __name__ == '__main__':
    optList, argv = getopt(sys.argv[1:], 'hd:A:B:m:')
    if not optList:
        optList.append(('-h',None))

 
    # these two lists store strs for initizing parameters

    paramAList = []
    paramBList = []

    dataPath = ''
    mode = 0
    priceA = 1.
    priceB = 1.

    for opt, arg in optList:
        if opt == '-h':
            print '''
            \t\t d: dataPath. required.
            \t\t A: parameters for model A
            \t\t B: parameters for model B
            \t\t m: compare mode. 0: share training \t 1: not share training . Default: 0
            '''
        elif opt == '-d':
            dataPath = arg
    
        elif opt == '-A':
            paramAList.extend(arg.split())
            
        elif opt == '-B':
            paramBList.extend(arg.split())

        elif opt == '-m':
            mode = int(arg)

    if not dataPath:
        print 'data path need to be specified'

    paramA = Param(paramAList)
    paramB = Param(paramBList)

    learnerA = ftrl_proximal(paramA)
    learnerB = ftrl_proximal(paramB)

    recordA = ModelRecord()
    recordB = ModelRecord()

    aggregatorA = Aggregator() if paramA.aggregation else None
    aggregatorB = Aggregator() if paramB.aggregation else None

    print 'date:time\tImpA\tClkA\tSumLLA\tImpB\tClkB\tSumLLB'
    for date in xrange(20140701,20140731):
        for time in xrange(0,1440,10):
            if date == 20140710:
                continue
            else:
                fileStr = dataPath+'/'+str(date)+'/'+str(time)+'.txt.gz'
                for instance in dataGenerator(fileStr):
                    
                    xA, y = process(instance.copy(), paramA.D, aggregatorA)
                    pA = learnerA.predict(xA)
                    xB, y = process(instance.copy(), paramB.D, aggregatorB)
                    pB = learnerB.predict(xB)
                    
                    # share training data
                    if mode:    
                        # A wins
                        if pA > pB or (pA == pB and random() > 1./2):
                            recordA.update(pA, y)
                        else:
                            recordB.update(pB, y)

                        # both update
                        learnerA.update(xA, pA, y)
                        learnerB.update(xB, pB, y)
                        if aggregatorA:
                            aggregatorA.update(instance, y)
                        if aggregatorB:
                            aggregatorB.update(instance, y)

                    # not share training data
                    else:
                        if pA > pB or (pA == pB and random() > 1./2):
                            recordA.update(pA, y)
                            learnerA.update(xA, pA, y)
                            if aggregatorA:
                                aggregatorA.update(instance, y)
                        else:
                            recordB.update(pB, y)
                            learnerB.update(xB, pB, y)
                            if aggregatorB:
                                aggregatorB.update(instance, y)

                print '%d\t%d\t%f\t%d\t%d\t%f' % (recordA.imp, recordA.clk, recordA.sumLL, recordB.imp, recordB.clk, recordB.sumLL) 
