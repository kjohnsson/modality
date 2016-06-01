'''
    Importing table qDiptab from R package diptest and write to file
    qDiptab.
'''

import os
from rpy2 import robjects
from rpy2.robjects.packages import importr, data
from rpy2.rinterface import RRuntimeError


def main():
    try:
        diptest = importr('diptest')
    except RRuntimeError:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('diptest')
        diptest = importr('diptest')

    qDiptab = data(diptest).fetch('qDiptab')['qDiptab']
    fname_diptab = robjects.StrVector([os.path.join(os.path.dirname(__file__), 'qDiptab.csv')])
    robjects.r('write.csv({}, {})'.format(qDiptab.r_repr(), fname_diptab.r_repr()))

if __name__ == '__main__':
    main()
