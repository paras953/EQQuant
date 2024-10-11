import time

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        run_time = end-start
        print(f'Function {func.__name__} took {run_time:.2f} seconds')
        return result
    return wrapper