import multiprocessing
from tqdm import tqdm 
def myfunction(x):
    return x**2 
if __name__ == "__main__":
    l = []  
    for i in tqdm(range(400)):
        with multiprocessing.Pool() as p: 
            x= p.map(myfunction , range(5))
            l.append(x)
        print('yes!')
    print(l)
