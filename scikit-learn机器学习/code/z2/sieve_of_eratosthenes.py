import numpy as np

'''
Sieve of Eratosthenes打印1~100中的质数
'''

# 生成从1~100的数组
a = np.arange(1, 101)

# 该方法最多到开方位置即完成
n_max = int(np.sqrt(len(a)))
# 初始化所有数都设置为"是质数"
is_prime = np.ones(len(a), dtype=bool)
# 1不是质数,2是质数,已经默认设置成True
is_prime[0] = False

for i in range(2, n_max):
    # 如果i是质数
    if i in a[is_prime]:
        # 从i^2开始每次+i的数都因为能被i整除所以不是质数
        is_prime[(i ** 2 - 1)::i] = False
        # 这里使用从i^2开始,那么比i小的i的倍数都在前面的迭代中设置过为False了

print(a[is_prime])
