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


import gzip
import sys
import os
from getopt import getopt
from collections import deque, defaultdict
from mmh3 import hash, hash64
from datetime import datetime, timedelta
from csv import DictReader
from math import exp, log, sqrt


def _getmalltagmap():
    map = {}
    with open('mallId.txt') as mallData:
        for line in mallData:
            words = line.split()
            map[words[0]] = words[1]       

    return map


def _tofloat(str):
    if str=='':
        return 0
    else:
        return float(str)


class Aggregator(object):
    ''' This class record the last ad time based on attributes, e.g, ('mallId','cookieId').
        The attributes are hard coded in the constructor.
    '''


    def __init__(self, mem_len):
        # mem_len defines the max list length for each key. 
        self._counter_map = defaultdict(deque)
        self._keys = [('mallId','cookieId')]
        self.mem_len = mem_len

    def update(self,instance,y):
        for aggKey in self._keys:
            key_for_update = hash64(str(tuple([key+'_'+instance[key] for key in aggKey])))  # hash for memory issue
            temp_list = self._counter_map[key_for_update]
            if len(temp_list) == self.mem_len:
                temp_list.popleft()
            temp_list.append((self.time,y))

    def gen_features(self,instance,logtime,D):
        # generate features based on instance's attribute.
        # For each key, we generate hash((bin(logtime-time[i]),i,lastY[i])) % D
        for aggKey in self._keys:
            key_for_feature = hash64(str(tuple([key+'_'+instance[key] for key in aggKey])))
            for idx, content in enumerate(self._counter_map[key_for_feature]):
                time, lastY = content
                val = int(log((logtime - time).total_seconds() + 1.))
                yield abs(hash(str(aggKey)+'_'+str(idx)+'_'+str((val,lastY)))) % D , 1.


#############################################################################################
# Parameters ################################################################################
#############################################################################################

class Param(object):
    ''' This class has the following:
        1. FTRL-Proximal model parameters: alpha, beta, L1, L2
        2. Instance processing parameters (about feature engineering): D(hashing trick), interaction, aggregation
        3. Training / Validation parameters: epoch, detectTC(whether to track training cost)
    '''
    optstr = 'a:b:l:L:D:I:e:A:V'

    def __init__(self, opt_list):

        # model
        self.alpha = .1
        self.beta = 1.
        self.L1 = 1.
        self.L2 = 1.

        # feature/hash trick
        self.D = 2 ** 25
        self.interaction = 0         # 0: linear /  1: poly2 with linear / 2: poly2 without linear
        self.aggregation = 0     # whether to enable the aggregator

        # D, training/validation
        self.detectTC = False       # detect training logloss

        for opt, arg in opt_list:
            if opt == '-a':
                self.alpha = float(arg)

            elif opt == '-b':
                self.beta = float(arg)

            elif opt == '-l':
                self.L1 = float(arg)

            elif opt == '-L':
                self.L2 = float(arg)

            elif opt == '-I':
                self.interaction = int(arg)

            elif opt == '-D':
                base, pow = arg.split('**')
                self.D = int(base) ** int(pow)

            elif opt == '-A':
                self.aggregation = int(arg)


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class FtrlProximal(object):
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
        yield 0, 1.

        # then yield the linear indices
        if self.interaction != 2:
            for i,val in x:
                yield i,val

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i][0]) + '_' + str(x[j][0]))) % D, x[i][1]*x[j][1]

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
        for i,val in self._indices(x):
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

            wTx += w[i] * val


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
        for i, val in self._indices(x):
            g_i = g * val
            sigma = (sqrt(n[i] + g_i * g_i) - sqrt(n[i])) / alpha
            z[i] += g_i - sigma * w[i]
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


def datagenerator(filename):
    with gzip.open(filename) as datafile:
        datareader = DictReader(datafile,delimiter='\t')
        for instance in datareader:
            yield instance


def process(instance, D, aggregator=None, malltagmap=_getmalltagmap()):
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

    time_format = '%Y%m%d%H%M%S'

    # process target
    try:
        y = (0.0,1.0)[instance['y']=='1']
        del instance['y']

        # delete useless feature
        del instance['impToken']
        del instance['userSegment']
        site_category = instance['siteCategory']
        del instance['siteCategory']
        del instance['directDeal']
        del instance['adViewBeginTimeOfLastSession']

        # campaign should be a subcategory of mall / publisherChannel should be a subcateogory of publisher
        instance['campaign'] = instance['mallId'] + '_' + instance['campaign']
        instance['publisherChannel'] = instance['publisher'] + '_' + instance['publisherChannel']

        # feature Engineering

        ##  binning
        instance['visitSessions'] = int(2.5 * log(_tofloat(instance['visitSessions'])))
        instance['visitsOfLastSession'] = int(2.5 * log(_tofloat(instance['visitsOfLastSession'])))
        instance['maxVisitsOfSession'] = int(log(_tofloat(instance['maxVisitsOfSession'])))
        instance['buySessions'] = int(2.5 * _tofloat(instance['buySessions']) + 1)
        instance['adViewsOfLastSession'] = int(2.5 * log( _tofloat(instance['adViewsOfLastSession']) + 1))
        instance['adEffectiveViewsOfLastSession'] = int(2.5 * log(_tofloat(instance['adEffectiveViewsOfLastSession']) + 1))
        instance['adViewsSinceLastVisit'] = int(2.5 * log(_tofloat(instance['adViewsSinceLastVisit']) + 1))
        instance['adSessions'] = int(2.5 * log(_tofloat(instance['adSessions']) + 1))
        instance['decayedAdSessions'] = int(2.5 * log(_tofloat(instance['decayedAdSessions'] ) + 1))
        instance['adEffectiveViewsSinceLastVisit'] = int(2.5 * log(_tofloat(instance['adEffectiveViewsSinceLastVisit'] ) + 1))

        ## process time 
        logtime = datetime.strptime(instance['logTime'],time_format)
        lastvisittime = datetime.strptime(instance['lastVisitTime'],time_format)
        del instance['logTime']
        del instance['lastVisitTime']

        instance['logT'] = ( logtime.hour * 60 + logtime.minute) / 30
        instance['logD'] = logtime.day
        instance['logW'] = logtime.weekday()
        instance['lastT'] =  ( lastvisittime.hour * 60 + lastvisittime.minute ) / 30
        instance['lastD'] = lastvisittime.day
        instance['lastW'] = lastvisittime.weekday()
        instance['tdLogLast'] = int(log((logtime - lastvisittime).total_seconds() / 360.+ 1))

        ## lastBuySessionTime
        if instance['lastBuySessionTime']:
            lastbst = datetime.strptime(instance['lastBuySessionTime'],time_format)
            td_log_bst = logtime - lastbst
            td_last_bst = lastvisittime - lastbst

            instance['tdLogBST'] = int(log(td_log_bst.total_seconds() / 360. + 1))
            if td_last_bst.days < 0:
                instance['tdLastBST'] = -1
            else:
                instance['tdLastBST'] = int(log(td_last_bst.total_seconds() / 360. + 1))

        del instance['lastBuySessionTime']

        ## findPriceTagTime
        if instance['findPriceTagTime']:
            ft = datetime.strptime(instance['findPriceTagTime'],time_format)
            td_log_ft = logtime - ft
            td_last_ft = lastvisittime - ft
            instance['tdLogFt'] = int(log(td_log_ft.total_seconds() / 360. + 1))
            instance['tdLastFt'] =(-1,1)[td_last_ft.days >= 0]* (int( log( abs(td_last_ft.total_seconds()) / 360. + 1)) + 1)

        del instance['findPriceTagTime']


        # manually generated feature:  Generating features by myself!!!!!!!!!!!!!!!!!!!!!!!!

        if not malltagmap[instance['mallId']]:
            print "cannot find mall category. Mall ID:".format(instance['mallId'])

        instance['mallCategory'] = malltagmap[instance['mallId']]


        #build categorical features
        x = []

        for k, v in instance.iteritems():
            # one-hot encode everything with hash trick
            index = abs(hash(k + '_' + str(v))) % D
            x.append((index,1.))

        #build numerical features
        if site_category:
            for category in site_category.split('|'):

                words = category.split(':');
                try:
                    catId, val = words[0], float(words[1])
                except BaseException:
                    exit(0)

                x.append((abs(hash('siteCategory_'+catId)) % D, val))

        #build aggregator numerical feature
        if aggregator:
                
            for feature in aggregator.gen_features(instance,logtime,D):
                x.append(feature)
            
            aggregator.time = logtime
            

    except BaseException as error:
        print error
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        for k,v in instance.iteritems():
            print k,v
        exit(0)


    return x, y


##############################################################################
# start training #############################################################
##############################################################################

if __name__ == "__main__":
    optstr = Param.optstr + 'd:' + 'h'
    opt_list, argv =  getopt(sys.argv[1:],optstr)
    if not opt_list or [ opt for opt in opt_list if opt[0] == '-h']:
        print '''
                  \t\t-h\t help
                  \t\t-a\t learning rate. alpha. default = .1
                  \t\t-b\t learning rate. beta. default = 1.
                  \t\t-l\t L1 regularization. default = 1.
                  \t\t-L\t L2 regularization. default = 1.
                  \t\t-D\t feature dimensions for hash trick. format: a**b. default = 2**20
                  \t\t-A\t memory length for aggregator. Aggregator disabled if 0. default = 0
                  \t\t-d\t data path. Required.
              '''
        exit(0)

    for opt,arg in opt_list:
        if opt == '-d':
            datapath = arg

    param_opt_list = [opt for opt in opt_list if opt[0] != '-d']
    param = Param(param_opt_list)
    learner = FtrlProximal(param)

    # start training
    print 'date:time\telapsed time\tvalidation batch logloss\tvalidation online logloss'

    startTime = datetime.now()
    aggregator = Aggregator(param.aggregation) if param.aggregation else None

    for date in xrange(20140701,20140731):
        if date == 20140710:
            continue
        for time in xrange(0,1440,10):
            filename = os.path.join(datapath,str(date),str(time) + '.txt.gz')

            vallogloss = 0 # vallogloss sums the batch log loss 
            valcount = 0
            for instance in datagenerator(filename):
                x, y = process(instance, param.D, aggregator)
                vallogloss += logloss(learner.predict(x),y)
                valcount += 1

            valonlinelogloss = 0 # valonlinelogloss sums the online log loss

            for instance in datagenerator(filename):
                x, y = process(instance, param.D, aggregator)
                p = learner.predict(x)
                valonlinelogloss += logloss(p,y)
                learner.update(x,p,y)
                if aggregator:
                    aggregator.update(instance,y)

            print '{}:{}\t{}\t{}\t{}'.format( date,time,(datetime.now()-startTime).total_seconds(),vallogloss/valcount,valonlinelogloss/valcount)
