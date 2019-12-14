def fib_python(n):
    '''Returns the nth Fibonacci number.'''
    a, b = 0, 1
    for i in range(n):
        a, b = a + b, a
    return a


if __name__ == '__main__':
    print("##### check result #####")
    import fib
    print("fib(47) in python:", fib_python(47))
    print("fib.fib_c(47):", fib.fib_c(47))
    print("fib.fib_cython(47):", fib.fib_cython(47))
    print("fib.fib_cython_optimized(47):", fib.fib_cython_optimized(47))

    print("\n##### performace benchmark #####")
    import timeit
    python_setup = "from __main__ import fib_python"
    cython_setup = "import fib"
    print("Python code: ", timeit.timeit('fib_python(47)', setup=python_setup), "seconds")
    print("Cython code: ", timeit.timeit('fib.fib_cython(47)', setup=cython_setup), "seconds")
    print("Optimized Cython code: ", timeit.timeit('fib.fib_cython_optimized(47)', setup=cython_setup), "seconds")
    print("C code: ", timeit.timeit('fib.fib_c(47)', setup=cython_setup), "seconds")