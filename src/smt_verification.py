from z3 import *

def violated(gap, eps=0.01):
    return gap > eps

def verify_maxgap(gap, eps=0.01):

    s = Solver()

    g = Real("g")

    s.add(g == gap)
    s.add(g > eps)

    return s.check()
