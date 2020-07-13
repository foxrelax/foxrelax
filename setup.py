# -*- coding:utf-8 -*-
from setuptools import (setup, find_packages)
import foxrelax as relax

long_desc = """
FoxRelax
"""

setup(
    name='foxrelax',
    version=relax.version(),
    description='foxrelax',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='liqiang',
    author_email='liqiang@g.im',
    license='BSD License',
    platforms=['all'],
    url='https://github.com/foxrelax/foxrelax',
    install_requires=relax.install_requires(),
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
