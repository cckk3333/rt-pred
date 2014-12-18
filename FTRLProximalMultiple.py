'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime, timedelta
from csv import DictReader
from math import exp, log, sqrt
import sys 
import gzip
import os
import mmh3
import itertools
from getopt import getopt


# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


def myFloat(str):
    if str=='':
        return 0
    else:
        return float(str)

##############################################################################
# parameters #################################################################
##############################################################################

class Aggregator():
    
    # here: we aggregate publisherId, mallId, cookieId, 

    def __init__(self):
        self.counterMap = {}  # key : decay factor (unit: 10 mins ); value : corresponding counterDict
        self.counterMap[0.] = {}  # counter not decayed
        self.counterMap[1.] = {}  # counter with decay factor: half per 10 minutes
        self.counterMap[1./6] = {}  # counter with decay factor: half per hour
        self.counterMap[1./144] = {} # counter with decay factor: half per day

        self._keys = ['publisher','mallId','cookieId'] 
    
    def _indices(self):
        for r in xrange(len(self._keys)):
            for com in itertools.combinations(self._keys,r+1):
                yield com

    def update(self,instance,y):
        for aggKey in self._indices(): # aggKey is somthing like ('publisher'), ('publiserId, mallId') 
            keyForUpdate = tuple([key+'_'+instance[key] for key in aggKey]) # keyForUpdate is somthing like ('publisherId_123213'), ('publiserId_12312312,mallId_123')
            for decayFactor in self.counterMap:
                if keyForUpdate not in self.counterMap[decayFactor]:
                    self.counterMap[decayFactor][keyForUpdate] = [0.]*2 # (#clicks,#impressions)
                self.counterMap[decayFactor][keyForUpdate][0] = self.counterMap[decayFactor][keyForUpdate][0] + 1. 
                self.counterMap[decayFactor][keyForUpdate][1] = self.counterMap[decayFactor][keyForUpdate][1] + y == 1.
                

    
    def genFeatures(self,instance):
        # default: generate CTR feature 
        # Other features for future work: pseudo CTR, #click, #imp
        for aggKey in self._indices(): # aggKey is somthing like ('publisher'), ('publiserId, mallId') 
            keyForFeature = tuple([key+'_'+instance[key] for key in aggKey]) # keyForFeature is somthing like ('publisherId_123213'), ('publiserId_12312312,mallId_123')
            for decayFactor in self.counterMap:
                if keyForFeature not in self.counterMap[decayFactor]:
                    self.counterMap[decayFactor][keyForFeature] = [0.]*2 # (#clicks,#impressions)
                yield decayFactor,keyForFeature,self.counterMap[decayFactor][keyForFeature]
        


    def decay(self,instance):
        # Since each file has 10-minute data and we decay once for each file, the decay unit is 10 minutes.
        for decayFactor in counterMap:
            for key in counterMap[decayFactor]:
                self.counterMap[decayFactor][key][0] *= (1./2) ** dacayFactor
                self.counterMap[decayFactor][key][1] *= (1./2) ** decayFactor


class Param():
    def __init__(self):
        # B, model
        self.alpha = .1  # learning rate
        self.beta = 1.   # smoothing parameter for adaptive learning rate
        self.L1 = 1.     # L1 regularization, larger value means more regularized
        self.L2 = 1.     # L2 regularization, larger value means more regularized

        # C, feature/hash trick
        self.D = 2 ** 25             # number of weights to use
        self.interaction = False     # whether to enable poly2 feature interactions
        self.aggregation = False     # whether to enable the aggregator
        self.norm = False            # whether to normalize the feature vector length  ( x <- x / sqrt(len(x))))

        # D, training/validation
        self.epoch = 1       # learn training data for N pass
        self.vmode = 1       # validation mode. 1: bin 2: online
        

    def __str__(self):
        str = ("alpha:{}\n".format(self.alpha)
        +  "beta:{}\n".format(self.beta) 
        +  "L1:{}\n".format(self.L1) 
        +  "L2:{}\n".format(self.L2)
        +  "D:{}\n".format(self.D) 
        +  "interaction:{}\n".format(self.interaction) 
        +  "aggregation:{}\n".format(self.aggregation))
        
        return str

        
        
        

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, param):
        # parameters
        self.alpha = param.alpha
        self.beta = param.beta
        self.L1 = param.L1
        self.L2 = param.L2

        # feature related parameters
        self.D = param.D
        self.interaction = param.interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * param.D
        self.z = [0.] * param.D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x
        
            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        ''' 

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]


        # normalization 
        if (param.norm):
            wTx = wTx / sqrt(len(x))

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y
    
        # update z and n
        normConst = (1. / sqrt(len(x)) , 1. ) [not param.norm]
        g_i = g * normConst
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g_i * g_i) - sqrt(n[i])) / alpha
            z[i] += normConst * g_i - sigma * w[i]
            n[i] += g_i * g_i


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)



def data(file, D, aggregator, aggregationFlag):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    timeFormat = '%Y%m%d%H%M%S'
    with gzip.open(file) as dataFile:
        dataReader = DictReader(dataFile,delimiter='\t')
        for t, instance in enumerate(dataReader):
            # process target
            try:
                y = (0.0,1.0)[instance['y']=='1']
                del instance['y']

                # delete useless feature
                del instance['impToken']
                del instance['userSegment']
                del instance['campaign']
                del instance['publisherChannel']
                del instance['siteCategory']
                del instance['directDeal']
                del instance['adViewBeginTimeOfLastSession']


                # feature Engineering
                logTime = datetime.strptime(instance['logTime'],timeFormat)
                lastVisitTime = datetime.strptime(instance['lastVisitTime'],timeFormat)
                del instance['logTime']
                del instance['lastVisitTime']
                
                instance['logT'] = ( logTime.hour * 60 + logTime.minute) / 30
                instance['logD'] = logTime.day
                instance['logW'] = logTime.weekday()
                instance['lastT'] =  ( lastVisitTime.hour * 60 + lastVisitTime.minute ) / 30
                instance['lastD'] = lastVisitTime.day
                instance['lastW'] = lastVisitTime.weekday()
                instance['tdLogLast'] = int(log((logTime - lastVisitTime).total_seconds() / 3600.0 + 1))

                ##  binning
                instance['visitSessions'] = int(2.5 * log(myFloat(instance['visitSessions']))) 
                instance['visitsOfLastSession'] = int(2.5 * log(myFloat(instance['visitsOfLastSession'])))
                instance['maxVisitsOfSession'] = int(log(myFloat(instance['maxVisitsOfSession'])))
                instance['buySessions'] = int(2.5 * myFloat(instance['buySessions']) + 1)
                instance['adViewsOfLastSession'] = int(2.5 * log( myFloat(instance['adViewsOfLastSession']) + 1))
                instance['adEffectiveViewsOfLastSession'] = int(2.5 * log(myFloat(instance['adEffectiveViewsOfLastSession']) + 1))
                instance['adViewsSinceLastVisit'] = int(2.5 * log(myFloat(instance['adViewsSinceLastVisit']) + 1))                    
                instance['adSessions'] = int(2.5 * log(myFloat(instance['adSessions']) + 1))
                instance['decayedAdSessions'] = int(2.5 * log(myFloat(instance['decayedAdSessions'] ) + 1))
                instance['adEffectiveViewsSinceLastVisit'] = int(2.5 * log(myFloat(instance['adEffectiveViewsSinceLastVisit'] ) + 1))

                
                ## lastBuySessionTime
                if instance['lastBuySessionTime']:
                    lastBST = datetime.strptime(instance['lastBuySessionTime'],timeFormat)
                    tdLogBST = logTime - lastBST
                    tdLastBST = lastVisitTime - lastBST
                    
                    instance['tdLogBST'] = int(log(tdLogBST.total_seconds() / 3600.0 + 1))
                    if tdLastBST.days < 0:
                        instance['tdLastBST'] = -1
                    else:
                        instance['tdLastBST'] = int(log(tdLastBST.total_seconds() / 3600.0 + 1))
               
                    del instance['lastBuySessionTime']
                
                ## findPriceTagTime
                if instance['findPriceTagTime']:
                    ft = datetime.strptime(instance['findPriceTagTime'],timeFormat)
                    tdLogFt = logTime - ft
                    tdLastFt = lastVisitTime - ft
                    instance['tdLogFt'] = int(log(tdLogFt.total_seconds() / 3600.0 + 1))
                    instance['tdLastFt'] =(-1,1)[tdLastFt.days >= 0]* (int( log( abs(tdLastFt.total_seconds()) / 60 + 1)) + 1)
                    del instance['findPriceTagTime']


                #build x
                x = []

                for k, v in instance.iteritems():
                    # one-hot encode everything with hash trick
                    index = abs(mmh3.hash(k + '_' + str(v))) % D
                    x.append(index)

                #add aggregator features
                if aggregationFlag:
                    for df, key, stats in aggregator.genFeatures(instance):
                        ctr = int( 5000 * stats[0] / ( stats[1] + 1. ))  # just handle the divide zero error
                        index = mmh3.hash(str(df)+str(key)+str(ctr)) % D
                        x.append(index)


            except BaseException as error:
                print error
                print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
                for k,v in instance.iteritems():
                    print k,v
                exit(0)

            
            yield t, x, y, instance


##############################################################################
# start training #############################################################
##############################################################################

if __name__ == "__main__":

    param = Param()
    dataPath=''    
    requiredOptList = ['-d']
    optList,argv = getopt(sys.argv[1:],'ha:b:l:L:D:Ie:d:v:AN')
    if not optList:
        optList.append(('-h', None))
    
    # print optList

    for opt, arg in optList:
        if opt == '-h':
            print '''
            \t\t -h\t  see help
            \t\t -a\t  alpha for learning rate. default a = .1
            \t\t -b\t  smooth parameter for adaptive learning rate. default b = 1.
            \t\t -l\t  L1 regularization. default l = 1
            \t\t -L\t  L2 regularization. default L = 1
            \t\t -D\t  #features for hashing trick. format: a**b. default D = 2**20
            \t\t -I\t  enable interaction term.
            \t\t -A\t  enable aggregation term.
            \t\t -e\t  epoch. deault e = 1
            \t\t -d\t  Required. data directory. 
            \t\t -n\t  enable feature vector normalization.  x = x / | x |
            '''
            exit()

        elif opt == '-a':
            param.alpha = float(arg)
        
        elif opt == '-b':
            param.beta = float(arg)
        
        elif opt == '-l':
            param.L1 = float(arg)

        elif opt == '-L':
            param.L2 = float(arg)

        elif opt == '-I':
            param.interaction = True

        elif opt == '-D':
            base, pow = arg.split('**')
            param.D = int(base) ** int(pow)
    
        elif opt == '-e':
            param.epoch = int(arg)
        
        elif opt == '-d':
            dataPath = arg

        elif opt == '-v':
            param.vmode = int(arg)

        elif opt == '-A':
            param.aggregation = True

        else:
            print ('oops!unknown parameter {}' % opt)
            exit(0)

    for opt in requiredOptList:
        if opt not in [opt[0] for opt in optList]:
            print ("{} is required." % opt)
    
    # initialize ourselves a learner
    learner = ftrl_proximal(param)

    # start training
    print 'date:time\telapsed time\tvalidation logloss\ttraining logloss'
    
    startTime = datetime.now()
    aggregator = Aggregator()

    for date in xrange(20140701,20140731):
        if date == 20140710:
            continue
        for time in xrange(0,1440,10):
            file = os.path.join(dataPath,str(date),str(time) + '.txt.gz')
            # count log loss
            valLogLoss = 0 
            valCount = 0
            for t, x, y, instance in data( file, param.D, aggregator, param.aggregation):
                # default compute log loss without sampling
                valLogLoss += logloss(learner.predict(x),y)
                valCount += 1
            
            trainLogLossList =[]
            for i in xrange(param.epoch):
                trainLogLoss = 0
                for t, x, y, instance in data( file, param.D, aggregator, param.aggregation):
                    p = learner.predict(x)
                    trainLogLoss += logloss(p,y)
                    learner.update(x,p,y)
                    if (param.aggregation):
                        aggregator.update(instance,y)
                
                trainLogLossList.append(trainLogLoss/valCount)

            
            print '{}:{}\t{}\t{}\t{}'.format( date,time,(datetime.now()-startTime).total_seconds(),valLogLoss/valCount,trainLogLossList)

    

