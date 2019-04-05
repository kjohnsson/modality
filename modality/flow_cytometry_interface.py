import numpy as np

from .resampling_tests import calibrated_diptest, calibrated_bwtest, silverman_bwtest
from .diptest import hartigan_diptest
from .excess_mass_modes import excess_mass_modes
from .util import fp_blurring


def flow_cytometry_interface(fun):
    """
    Adds preprocessing of data to a function for testing for
    unimodality. The new function will have additional keyword
    arguments:

        ch_range    -   tuple with min and max value of channel
        rm_extreme  -   should data at the maximum and minimum
                        values be removed?
        blurring    -   type of blurring: 'standard', 'fp',
                        'fp_deterministic' or 'none'
        blur_delta  -   discretization step for truncated data (float)
                        or 'infer', which infers the step.
    """

    def fun_with_fc_interface(*args, **kwargs):

        preproc = {'ch_range': None, 'blurring': 'fp', 'blur_delta': 'infer',
                   'rm_extreme': True}
        try:
            data = args[0]
        except IndexError:
            data = kwargs.pop('data')
        for key in preproc:
            if key in kwargs:
                preproc[key] = kwargs.pop(key)

        data = np.asarray(data)

        if preproc['ch_range'] is None:
            preproc['ch_range'] = np.min(data), np.max(data)
        if not ('I' in kwargs and kwargs['I'] != 'auto'):
                xmin, xmax = preproc['ch_range']
                kwargs['I'] = {'type': 'auto', 'xmin': xmin, 'xmax': xmax}

        data = preprocess_fcdata(data, **preproc)
        try:
            return fun(data, *args[1:], **kwargs)
        except TypeError as e:
            if "got multiple values for keyword argument 'I'" in str(e):
                raise TypeError("'I' can only be given as keyword argument"
                                "in flow cytometry interface.")
            if "got an unexpected keyword argument 'I'" in str(e):
                del kwargs['I']

        return fun(data, *args[1:], **kwargs)

    return fun_with_fc_interface


def infer_blur_delta(data):
    data = np.asarray(data)
    data_un = np.unique(data)
    if len(data_un) == len(data):
        raise ValueError("Data is not detectably truncated, select "
                         "'blurring=none' or set 'blur_delta' to numerical "
                         "value.")
    diffs = np.diff(np.sort(data_un))
    delta0 = np.min(diffs)
    return np.mean(diffs[diffs < 1.5*delta0])


def preprocess_fcdata(data, ch_range=None, rm_extreme=True, blurring='fp',
                      blur_delta='infer', return_list=False):
    """
    Blur data as preprocessing for tests for unimodality.
    Input:

        data        -   data set
        rm_extreme  -   should data at the maximum and minimum
                        values be removed?
        blurring    -   type of blurring: 'standard', 'fp',
                        'fp_deterministic' or 'none'
        blur_delta  -   discretization step for truncated data (float)
                        or 'infer', which infers the step.
    """

    data = np.asarray(data)

    if ch_range is None:
        ch_range = np.min(data), np.max(data)

    if rm_extreme:
        data = data[(data > ch_range[0]) & (data < ch_range[1])]

    if blur_delta == 'infer' and blurring != 'none':
        blur_delta = infer_blur_delta(data)

    blur_fun = {'standard': lambda data: data + (np.random.rand(*data.shape)-.5)*blur_delta,
                'fp': lambda data: fp_blurring(data, blur_delta, even_spaced=False),
                'fp_deterministic': lambda data: fp_blurring(data, blur_delta, even_spaced=True),
                'none': lambda data: data}
    blurred_data = blur_fun[blurring](data)

    if return_list:
        return blurred_data.tolist()

    return blurred_data


calibrated_diptest_fc = flow_cytometry_interface(calibrated_diptest)
calibrated_bwtest_fc = flow_cytometry_interface(calibrated_bwtest)
silverman_bwtest_fc = flow_cytometry_interface(silverman_bwtest)
hartigan_diptest_fc = flow_cytometry_interface(hartigan_diptest)
excess_mass_modes_fc = flow_cytometry_interface(excess_mass_modes)
