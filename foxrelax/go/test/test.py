# -*- coding:utf-8 -*-
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from foxrelax.go.test.agent_helper_test import EyeTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(EyeTest('test_corner'))
    suite.addTest(EyeTest('test_corner_false_eye'))
    suite.addTest(EyeTest('test_middle'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())