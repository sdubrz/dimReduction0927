# 循环执行，一次执行多个数据
from Run import run190927


def loop_run():
    # digit_count = [461, 526, 466, 477, 455, 421, 459, 487, 455, 464]
    digit_count = [863, 985, 874, 893, 853, 790, 860, 912, 854, 870]
    for i in range(0, 10):
        data_name = "MNIST50mclass" + str(i) + "_" + str(digit_count[i])
        run190927.run_test(data_name0=data_name)
        print("###############################################################")


if __name__ == '__main__':
    loop_run()
