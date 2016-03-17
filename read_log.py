import re
import glob

files = glob.glob('modality_compute_*')

for logfile in files:
    print "logfile = {}".format(logfile)

    with open(logfile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if re.match('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z]+) with null hypothesis ([a-z]+) at alpha = ([0-9.]+)$', line) is not None:
            test, null, alpha = \
                re.sub('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z]+) with null hypothesis ([a-z]+) at alpha = ([0-9.]+)$',
                       '\\3 \\4 \\5', line).split()

    print "Test: {}, null hypthesis: {}, at alpha = {}".format(test, null, alpha)

    i = len(lines)-1
    while i > 0:
        if 'gamma = {}'.format(alpha) in lines[i]:
            mean_val = re.sub('np.mean\(vals\) = ([0-9]+)', '\\1', lines[i+1])
            mean_val = float(mean_val)
            nbr_tests = re.sub('len\(vals\) = ([0-9]+)', '\\1', lines[i+2])
            nbr_tests = int(nbr_tests)
            break
        i -= 1

    while i > 0:
        if "Testing" in lines[i]:
            val = re.sub('Testing if ([0-9.]+) is upper bound for lambda_alpha', '\\1', lines[i])
            val = float(val)
            break
        i -= 1

    print "mean_val = {}".format(mean_val)
    print "Not able to decide for bound {} after {} tests".format(val, nbr_tests)
    print '-'*50