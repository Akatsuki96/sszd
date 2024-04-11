
from setuptools import setup, find_packages

setup(
    name="sszd",
    version="1.0.0",
    description="SSZD: Structured Stochastic Zero-th Order Descent",
    python_requires='>=3.9',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['numpy>=1.26', 'tqdm', 'matplotlib', 'pandas'],
    include_package_data=True,
)