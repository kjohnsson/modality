import re


def summarize_log(logfile, interval_search=True):
    print "logfile = {}".format(logfile)

    with open(logfile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if re.match('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z_]+) with null hypothesis ([a-z_0-9.]+) at alpha = ([0-9.]+)$', line) is not None:
            test, null, alpha = \
                re.sub('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z_]+) with null hypothesis ([a-z_0-9.]+) at alpha = ([0-9.]+)$',
                       '\\3 \\4 \\5', line).split()
            break

    try:
        print "Test: {}, null hypthesis: {}, at alpha={}".format(test, null, alpha)
    except:
        print "Test not found."
        return

    val = 0
    ntests = 0
    i = len(lines)-1
    while i > 0:

        bound_found = False
        max_samp_reached = 0
        gamma_str = '00: gamma_lower' if interval_search else 'gamma = {}'.format(alpha)
        while not (gamma_str in lines[i]) and i > 0:
            if '---' in lines[i]:
                ntests += 1
            if "Saved" in lines[i]:
                bound_type = re.sub('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z_]+) with null hypothesis ([a-z]+) at alpha = ([0-9.]+)$', '\\2', lines[i])
                bound_type = bound_type.replace('\n', '')
                val = re.sub('^Saved ([0-9.]+) as (upper|lower) bound for test ([a-z_]+) with null hypothesis ([a-z_0-9.]+) at alpha = ([0-9.]+)$', '\\1', lines[i])
                val = float(val)
                bound_found = True
                break
            i -= 1

        while i > 0:
            if gamma_str in lines[i]:
                mean_val = re.sub('^([0-9]+: ?)?np.mean\(vals\) = ([0-9]+)', '\\2', lines[i+1])
                mean_val = float(mean_val)
                nbr_tests = re.sub('^([0-9]+: ?)?len\(vals\) = ([0-9]+)', '\\2', lines[i+2])
                nbr_tests = int(nbr_tests)
                break
            i -= 1

        while i > 0:
            if 'max_samp' in lines[i]:
                max_samp_reached += 1
            if "Testing" in lines[i]:
                if interval_search:
                    val_new = re.sub('^.*Testing lambda_alpha = ([0-9.]+)$', '\\1', lines[i])
                else:
                    val_new = re.sub('^.*Testing if ([0-9.]+) is upper bound for lambda_alpha.*$', '\\1', lines[i])
                val_new = float(val_new)
                if not bound_found and val_new == val:
                    bound_found = True
                val = val_new
                break
            i -= 1

        if i > 0:
            if not bound_found:
                print "Not able to decide for bound {} after {} tests (mean val: {}, max samp reached: {}, nbr additional tests: {})".format(val, nbr_tests, mean_val, max_samp_reached, ntests)
            else:
                print "Decided for {} bound {} after {} tests (mean val: {}, max samp reached: {})".format(bound_type, val, nbr_tests, mean_val, max_samp_reached)

    print '-'*50
