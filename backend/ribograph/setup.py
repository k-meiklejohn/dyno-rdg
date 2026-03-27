from setuptools import find_packages, setup

setup(
    name='ribograph',
    packages=find_packages(include=['ribograph']),
    version='0.1.0',
    description='RiboGraph objects and related functions for generation of dynamic RDGs'
    author='Kyle Meiklejohn'
    install_requires=['networkx']
)