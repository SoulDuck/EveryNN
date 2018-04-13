#-*- coding:utf-8 -*-



class A(object):
    class_a='class a 3.'
    @classmethod
    def print_(cls):
        print 'class method 을 통해 클래스 변수를 불러옵니다 ', cls.class_a
        global class_a2
        cls.class_a_2 = 'class a_2'
        return 'class method 을 통해 클래스 변수를 불러옵니다 ' + cls.class_a
    @classmethod
    def change(cls):
        cls.class_a='CLASS A 3.'


    def __init__(self):
        print 'a'
        self.a=3


class B(A):
    def __init__(self):
        super(B,self).__init__()
        print 'Class B : ',self.a


class C(A):
    def __init__(self):
        super(C, self).__init__();

class D(A):
    def __init__(self):
        print '#',A.class_a
    def _print(self):
        print '##', A.class_a
    pass;


a=A()
print a.class_a
a.change()
print a.class_a


class_B=B()
class_B.print_()
class_C=C()
print class_B.class_a
print class_C.class_a
d=D()
d._print()
print d.class_a
"""
class A(object):
    def __init__(self):
        print "생성자 A"

class B(A):
    def __init__(self):
        super(B,self).__init__()
        print "생성자 B"

class C(B):
    def __init__(self):
        super(C,self).__init__()
        print "생성자 C"
"""

