---

title: "Quickly parallelize your python code"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Quickly parallelize your python code

---

# Quickly parallelize your python code

The need for parallelization usually occurs in the case of a long for loop, in this case, a few steps are to be taken in order to parallelize the process say we have a function:

```python
def add_numbers(end_in: int, start_from: int, print_done: bool) -> int:

    cur_sum = 0
    for i in range(start_from, end_in[1]):
        cur_sum += i

    if print_done:
        print(f'Run #{end_in[0]}, sum is: {cur_sum}')

    return cur_sum
```

This just sums some random numbers, say we want to use this function and sum really large numbers, the naïve implementation will be: 

### Naïve solution  - using for

```python
def use_for(iterable: Tuple[int, int]):

    total_sum=[]
    for (run_num, end_in) in iterable:

        total_sum.append(
                add_numbers((run_num, end_in), start_from=0, print_done=True))

    return sum(total_sum)
```

which is called by:

```python
    numbers_to_end_in = np.random.randint(low=1e8, size=10)
    iterable = zip(range(1,len(numbers_to_end_in)+1),numbers_to_end_in)

    t0=time.time()
    cur_sum = use_for(iterable)
    print(f'Time took using for is {time.time()-t0}\n')
    
# output: 
# Run #1, sum is: 4525173573611328
# Run #2, sum is: 305855759752003
# Run #3, sum is: 376915103136496
# Run #4, sum is: 3099323876907286
# Run #5, sum is: 15566528834850
# Run #6, sum is: 2728276306701
# Run #7, sum is: 26654100226881
# Run #8, sum is: 4007358828638490
# Run #9, sum is: 3191354287616440
# Run #10, sum is: 933400367456805

# Time took using for is 19.355337619781494 sum is 16484330702487280
```

This only takes some 18 sec but can escalate pretty quickly.

In case the script gets too slow, speeding things up can be done in a breeze:

### Parallel solution - using multiprocessing

Multiprocessing is most easily implemented using the `joblib`package.  

In order to take add_numbers and use it with map, we will have to define all the values we wish to remain constant via the `partial` function (imported from `functools`), this can be seen in line 3 (note that the non-constant values should be at the beginning of the function).

Once `add_numbers`'s partial counterpart is ready we call the `joblib` `Parallel` object. Below is the implemented function:

```python
from functools import partial
from joblib import Parallel, delayed

def use_parallel(iterable: Tuple[int, int]):

    par_add_numbers = partial(add_numbers, start_from=0, print_done=True)
    total_sum = Parallel(n_jobs=4)(delayed(par_add_numbers)(i) for i in iterable)
    return sum(total_sum)
```

Running the same script as before using this function, results in the following print:

```python
# output: 
# Run #2, sum is: 305855759752003
# Run #3, sum is: 376915103136496
# Run #6, sum is: 2728276306701
# Run #5, sum is: 15566528834850
# Run #7, sum is: 26654100226881
# Run #4, sum is: 3099323876907286
# Run #1, sum is: 4525173573611328
# Run #9, sum is: 3191354287616440
# Run #8, sum is: 4007358828638490
# Run #10, sum is: 933400367456805

# Time took using pmap is 5.731021165847778 sum is 16484330702487280
```

Its 3.4x faster. which is nice given the effort.

### A few notes

**Note!** map outputs a *generator*, it will only really operate if this generator is read.

Note!** debugging with multiprocessing is hard, it is however easy to have a debug flag that decides whether to use `map()` or `p.map()` 

### The full code

```python
import numpy as np
import time
from functools import partial
from joblib import Parallel, delayed
from typing import Tuple


def add_numbers(end_in: int, start_from: int, print_done: bool) -> int:

    cur_sum = 0
    for i in range(start_from, end_in[1]):
        cur_sum += i

    if print_done:
        print(f'Run #{end_in[0]}, sum is: {cur_sum}')

    return cur_sum


def use_for(iterable: Tuple[int, int]):

    total_sum=[]
    for (run_num, end_in) in iterable:

        total_sum.append(
                add_numbers((run_num, end_in), start_from=0, print_done=True))

    return sum(total_sum)


def use_parallel(iterable: Tuple[int, int]):

    par_add_numbers = partial(add_numbers, start_from=0, print_done=True)
    total_sum = Parallel(n_jobs=4)(delayed(par_add_numbers)(i) for i in iterable)
    return sum(total_sum)


if __name__ == '__main__':

    numbers_to_end_in = np.random.randint(low=1e8, size=10)

    iterable = zip(range(1,len(numbers_to_end_in)+1),numbers_to_end_in)
    
        t0=time.time()
    cur_sum = use_for(iterable)
    print(f'Time took using for is {time.time()-t0} sum is {cur_sum}\n')
    
    t0=time.time()
    cur_sum = use_parallel(iterable)
    print(f'Time took using pmap is {time.time()-t0} sum is {cur_sum}\n')
```

It's output on my MacBook pro i7:

```python
Run #1, sum is: 1102609928786796
Run #2, sum is: 1273806205416381
Run #3, sum is: 653951569179096
Run #4, sum is: 1443332470687278
Run #5, sum is: 65086981696890
Run #6, sum is: 508219372959066
Run #7, sum is: 4655369387408365
Run #8, sum is: 3721948669533276
Run #9, sum is: 82426258149795
Run #10, sum is: 1035466164356253
Time took using for is 22.911070346832275 sum is 14542217008173196

Time took using parallel is 5.906735897064209 sum is 14542217008173196
```

Getting almost `x4` speedup in a few lines of code.
