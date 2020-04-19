import ray
import time

ray.init()

@ray.remote
def f(x):
    return x ** 100

if __name__ == '__main__':
    start = time.time()
    ray.get([f.remote(i) for i in range(1000)])
    end = time.time()
    print(end - start)
