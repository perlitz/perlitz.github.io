---

title: "Quickly profile your Python code"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Quickly profile your Python code
---

# Quickly profile your Python code

## Profiling using a decorator:

Given some Python function for which profiling is required, add this decorator definition to the script:

```python
def profile(profiled_func):
    
    def wrap(*args, **kwargs):
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

        result = profiled_func(*args, **kwargs)

        pr.disable()
        pr.print_stats(sort='time')
        return result
    
    return wrap
```

Then, upon the definition of the function one wants to profile, just decorate it:

```python
@profile
def some_fun():
	pass
```

Upon running the script, the output will include the following:

- **ncalls** is the number of calls made.
- **tottime** is a total of the time spent in the function/method excluding the time spent in the functions/methods that it calls.
- **percall** refers to the quotient of tottime divided by ncalls
- **cumtime** is the cumulative time spent in this and all subfunctions. It’s even accurate for recursive functions!
- The second **percall column** is the quotient of cumtime divided by primitive calls
- **filename:lineno(function)** provides the respective data of each function