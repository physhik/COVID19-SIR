import time 

def f(x):
    return x**100

if __name__ == '__main__':
        start = time.time()
        map(f, range(10000))
        end = time.time()
        print(end - start)
