"""
PyGreedy Setup Script
==================

Setup script for installing PyGreedy package.

Created by: devhliu
Created at: 2025-02-18 05:22:28 UTC
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='pygreedy',
    version='0.1.0',
    author='devhliu',
    author_email='devhliu@example.com',
    description='A flexible medical image registration package using greedy algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/devhliu/pygreedy',
    packages=find_packages(include=['pygreedy', 'pygreedy.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'pygreedy=pygreedy.cli.main:main',
        ],
    },
    package_data={
        'pygreedy': [
            'config/*.json',
            'data/*.nii.gz',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'coverage>=5.5',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'black>=21.6b0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.2',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
    },
)