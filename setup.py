import os
from setuptools import setup, find_packages

import turingpoint


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='turingpoint',
    version=turingpoint.__version__,
    description="Reinforcement Learning (RL) library",
    author="Oren Zeev-Ben-Mordehai",
    author_email='zbenmo@gmail.com',
    url="https://github.com/zbenmo/emmerald",
    maintainer='zbenmo@gmail.com',
    packages=find_packages(),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=[
        'pandas>=1.5.2, <1.6',
        'numpy>=1.23.5, <1.24',
    ],
    extras_require={
        'examples': [
            'scikit-learn>=1.1.3, < 1.2',
            'gym >= 0.21, < 0.22',
            'stable_baselines3[extra]',
            'tqdm',
        ],
        'tests': [
            'pytest'
        ]
    },
    license=read("LICENSE"),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)