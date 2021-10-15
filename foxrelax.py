#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
from inspect import isfunction

# 优先从当前'开发目录'搜索foxrelax
# 注意:
# 如果系统已经安装了foxrelax, 仍然优先使用当前'开发目录'下的foxrelax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint:disable=wrong-import-position
import click


def root_path():
    return os.path.dirname(os.path.abspath(__file__))


def walk_path(path, callback=None):
    """
    递归的遍历path, 针对每个path调用callback(path)

    Notes
    -----
    def callback(path):
        pass

    Parameters
    ----------
    path : str
    callback : function

    Returns
    -------
    None

    """

    if not path:
        path = root_path()
    else:
        path = os.path.abspath(path)

    if isfunction(callback):
        callback(path)
    else:
        print(path)

    fnames = os.listdir(path)
    for fname in fnames:
        fpath = os.path.join(path, fname)
        if os.path.isdir(fpath):
            walk_path(fpath, callback)


@click.group()
def cli():
    pass


@cli.command()
def pylint():
    """
    pylint
    """

    rpath = root_path()

    def pylint_path(path):
        target_path = os.path.join(rpath, path)
        print(target_path)
        cmd = 'pylint {0}'.format(target_path)
        print(cmd)
        result = os.system(cmd)
        if result != 0:
            print('{0} error, code={1}'.format(cmd, result))

    pylint_path('foxrelax')
    pylint_path('setup.py')


@cli.command()
def yapf():
    """
    yapf
    """

    rpath = root_path()

    def callback(path):
        if os.path.exists(os.path.join(path, '__init__.py')):
            os.chdir(path)
            print(path)
            cmd = 'yapf -i *.py'
            print(cmd)
            result = os.system(cmd)
            if result != 0:
                print('{0} error, code={1}'.format(cmd, result))

    walk_path(os.path.join(rpath, 'foxrelax'), callback)


if __name__ == '__main__':
    cli()
