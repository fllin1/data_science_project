from setuptools import find_packages, setup
import sys
if sys.version_info[0:2] != (3, 11):
    raise Exception('Requires python 3.11')


setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Project for the "Put in Production a Data Science Project"',
    author='Florent LIN/Arthur SABRE/Alban DEREPAS',
    license='',
)
