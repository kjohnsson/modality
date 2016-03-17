import re

from src.calibration.lambda_alphas_access import save_lambda

savelog = '''
Saved 0.9375 as upper bound for test bw with null hypothesis fm at alpha = 0.05
Saved 0.90625 as lower bound for test bw with null hypothesis fm at alpha = 0.05

Saved 1.03125 as upper bound for test bw with null hypothesis shoulder at alpha = 0.1
Saved 1.0 as lower bound for test bw with null hypothesis shoulder at alpha = 0.1
'''

loglines = savelog.split('\n')

for line in loglines:
    if len(line) > 0:
        val, test, null, alpha = \
            re.sub('Saved ([0-9.]+) as (upper|lower) bound for test ([a-z]+) with null hypothesis ([a-z]+) at alpha = ([0-9.]+)',
                   '\\1 \\3 \\4 \\5', line).split()

        val = float(val)
        alpha = float(alpha)
        upper = 'upper' in line
        # print "val = {}".format(val)
        # print "alpha = {}".format(alpha)
        # print "test = {}".format(test)
        # print "null = {}".format(null)
        # print "upper = {}".format(upper)
        save_lambda(val, test, null, alpha, upper)
