import sys
from getopt import getopt
from FTRLProximal import *

if __name__ == '__main__':
    optList, argv = getopt(sys.argv[1:], 'hd:A:B:')
    if argv:
        print ('unrecognized parameters:' % argv)

    if not optList:
        optList.append(('-h',None))

 
    # these two lists store strs for initizing parameters

    paramAList = []
    paramBList = []

    dataPath = ''

    for opt, arg in optList:
        if opt == '-h':
            print '''
            \t\t h: help
            \t\t d: dataPath. required.
            \t\t A: parameters for model A
            \t\t B: parameters for model B
            '''
        elif opt == '-d':
            dataPath = arg
    
        elif opt == '-A':
            print arg
            paramAList.extend(arg.split())
            print paramAList

        elif opt == '-B':
            paramBList.extend(arg.split())

    if not dataPath:
        print 'data path need to be specified'

    paramA = Param(paramAList)
    paramB = Param(paramBList)

    
