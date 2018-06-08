def _fn(*args):
    for arg in args:
        print arg
        print len(args)
_fn((1,3) , (2,4))


print 30%100