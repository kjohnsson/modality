import sys
import traceback

from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['matplotlib', 'mpi4py', 'numpy', 'pandas',
                     'scikit-learn', 'scipy', 'six']

modality_data = ['data/gammaval.pkl']

if 'install' in sys.argv or 'develop' in sys.argv:
    try:
        from modality import write_qDiptab
        write_qDiptab.main()
        modality_data += ['data/qDiptab.csv']
        print("Tabluated p-values for Hartigan's diptest included.")
    except:
        traceback.print_exc()
        print("Tabulated p-values for Hartigan's diptest not loaded due to "
              "error. This means that p-values for Hartigan's "
              "(uncalibrated) diptest cannot be computed. Loading tabulated "
              "p-values requires the R package 'diptest' as well as the "
              "python module rpy2.")

print(find_packages())
setup(name='modality',
      version='1.1',
      description='Non-parametric tests for unimodality',
      author='Kerstin Johnsson',
      author_email='kerstin.johnsson@hotmail.com',
      url='https://github.com/kjohnsson/modality',
      license='MIT',
      packages=find_packages(),
      package_data={'modality': modality_data,
                    'modality.calibration': ['data/*.pkl']},
      install_requires=REQUIRED_PACKAGES,
      classifiers=[
          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6'
      ]
      )
