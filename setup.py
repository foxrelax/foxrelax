# -*- coding:utf-8 -*-
import os
import io
from setuptools import (setup, find_packages)

long_desc = """
FoxRelax
"""


def root_path():
    return os.path.dirname(os.path.abspath(__file__))


def version():
    fpath = os.path.join(root_path(), 'foxrelax', 'version.txt')
    with open(fpath, 'r') as f:
        f = open(fpath, 'r')
        line = f.readline()
        ver = line.strip()

    return ver


def install_requires():
    fpath = os.path.join(root_path(), 'requirements.txt')
    requires = []

    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                requires.append(line)

    return requires


setup(
    name='foxrelax',
    version=version(),
    description='foxrelax',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='liqiang',
    author_email='liqiang@g.im',
    license='BSD License',
    platforms=['all'],
    url='https://github.com/foxrelax/foxrelax',
    install_requires=install_requires(),
    keywords='foxrelax',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.txt']},
    classifiers=[
        'Development Status :: 1 - Planning', 'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries'
    ],
)
