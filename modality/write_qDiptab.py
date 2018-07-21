from __future__ import unicode_literals
'''
    Importing table qDiptab from R package diptest and write to file
    qDiptab.
'''
from __future__ import print_function

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

    qDiptab_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(qDiptab_dir):
        os.mkdir(qDiptab_dir)

    qDiptab = data(diptest).fetch('qDiptab')['qDiptab']
    fname_diptab = robjects.StrVector(
        [os.path.join(qDiptab_dir, 'qDiptab.csv')])
    robjects.r('write.csv({}, {})'.format(qDiptab.r_repr(), fname_diptab.r_repr()))
    print("Tabluated p-values loaded from R package diptest.")

if __name__ == '__main__':
    main()
