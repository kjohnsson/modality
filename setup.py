import sys
import src
from distutils.core import setup

setup(name='modality',
      version=src.__version__,
      description='Non-parametric tests for unimodality',
      author='Kerstin Johnsson',
      author_email='kerstin.johnsson@hotmail.com',
      url='https://github.com/kjohnsson/modality',
      license='MIT',
      packages=['modality', 'modality.calibration', 'modality.util'],
      package_dir={'modality': 'src'},
      package_data={'modality': ['data/*.csv'], 'modality.calibration': ['data/*.pkl']},
     )
