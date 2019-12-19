cdef extern from "cfib.h":
    unsigned long _fib "fib"(unsigned long n)

def fib_c(n):
    ''' Returns the nth Fibonacci number.'''
    return _fib(n)

def fib_cython(n):
    '''Returns the nth Fibonacci number.'''
    a, b = 0, 1
    for i in range(n):
        a, b = a + b, a
    return a

def fib_cython_optimized(unsigned long n):
    '''Returns the nth Fibonacci number.'''
    cdef unsigned long a=0, b=1, i
    for i in range(n):
        a, b = a + b, a
    return a