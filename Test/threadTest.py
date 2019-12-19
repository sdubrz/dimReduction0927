# 多线程测试

import _thread


def number_function(x):
    print(x*x)


def run_test():
    try:
        _thread.start_new_thread(number_function(2))
        _thread.start_new_thread(number_function(3))
        _thread.start_new_thread(number_function(4))
        _thread.start_new_thread(number_function(5))
    except:
        print("无法启动线程")
    # print(a)
    # print(b)


if __name__ == '__main__':
    run_test()

