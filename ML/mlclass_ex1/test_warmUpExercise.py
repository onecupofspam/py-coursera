import nose
from numpy import *
from numpy.testing import *

from exerciseOne import warmUpExercise

def testWarmUp():
    assert_array_equal(warmUpExercise(), eye(5))

if __name__ == '__main__':
    nose.runmodule()