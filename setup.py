
from setuptools import setup, find_packages

setup(
    name="stozhopt",
    version="0.0.0",
    description="Sto-ZhOpt: Stochastic Zero-th Order Optimization",
    python_requires='~=3.6',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['torch'],
    include_package_data=True,
)