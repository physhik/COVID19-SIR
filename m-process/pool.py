from multiprocessing import Pool
import time 
import gc

def f(x):
    return x**2

if __name__ == '__main__':
    with Pool() as p:
        gc.disable()
        start = time.time()
        p.map(f, range(10**8))
        end = time.time()
        print(end - start)
        gc.enable()
