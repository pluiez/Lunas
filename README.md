
# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.3.7-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python based library that mimics TensorFlow's dataset API to provide data processing pipeline 
and data feeding utility.

The implementation mostly draws on TensorFlow but in a simplified and pure-Python fashion. 

## Overview

A `Dataset` represents a dataset and holds corresponding pre-processing and filtering operations. 

Currently the following features are supported:

1. Dataset.repeat()
2. Dataset.interleave()
3. Dataset.shard()
4. Dataset.shuffle()
5. Dataset.sort()
6. Dataset.take()
7. Dataset.skip()


### Datasets

1. `Zip`: Zip multiple datasets.
2. `Concat`: Concat two datasets.
3. `Range`: Behave similarly to the builtin range().
4. `Enumerate`: Similar to the builtin enumerate().
5. `Array`: Represent an array as dataset. An array could be an iterable, iterator or numpy array.
6. `Glob`: Use Python's glob module to match potentially multiple files and wraps them into a dataset. 
7. `Stdin`: A dataset that reads from standard input.
8. `TextLine`: Represent a plain-text file.


### Iterator

An `Iterator` generates batches by iterating through the dataset.

Currently supported batching schemes include:

1. `SimpleIterator`: Used to batch according to number of samples.
2. `BucketIterator`: Arrange samples of similar length together. Usually it's used to reduce redundant 
computations and thus maximize GPU utilization.

These iterators can work with PyTorch's DataLoader to exploit parallelism for data processing. A convenient `DataLoader`
is provided for this purpose.


## Requirements

- setuptools
- typings
- numpy
- Python >= 3.7


## Installation

Install using pip:

```shell
pip install lunas
```

## Examples

1. Create a dataset and iterate through it

   ```python
   from lunas import Range

   ds = Range(10).shuffle(buffer_size=5)
   for x in ds: # epoch 1
       print(x)
   for x in ds: # epoch 2
       print(x)
   ```
    
    - A `Range` dataset is created similarly to `range(10)` and then iterated through for one epoch.
    As you see, we can scan through it for several times.

    - Instead of iterating twice, you can also use `ds.repeat(2)` to create another dataset that does the same work.
    
    - Dataset.shuffle() performs a buffered shuffling, which uses a queue under the hood.
   
2. Build a data processing pipeline

   ```python
   ds = Range(10).select(lambda x: x * 2).select(lambda x: x + 1).where(lambda x: x % 2 == 0)
   ```

   - The chaining calls of a `Dataset` object defines a processing pipeline on the dataset.
   
   - `select(fn)` applies transformation on every dataset element lazily. The argument `fn` is a custom 
   mapping function that takes a single sample as input and output. You can apply any transformation to a dataset and return a sample of any type, e.g., `Dict`, `List` or a custom `Sample`.
   
   - `where(fn)` accepts a predicate and returns a `bool` value to filter an input sample. If returned True, 
   the sample is preserved, otherwise discarded.
   
   - Both `select(fn)` and `where(fn)` returns the dataset itself just to enable chaining style invocations. The mapping and filtering ops are attached to the dataset immediately in an in-place fashion.

3. Deal with multiple input sources

   ```python
   from lunas import Range, Zip

   ds1 = Range(10)
   ds2 = Range(start=10, stop=20, step=1)
   ds = Zip([ds1, ds2]).select(lambda x,y: x + y)
   ```

   - We create two datasets and pack them as a `Zip` dataset. A `Zip` dataset returns a tuple from its internal datasets in the given order.
   
   - `Zip` also allows zip multiple datasets of different size by providing additional `mode` argument and `paddinng` to indicate the way for padding shorter dataset.
   

4. Practical use case in a distributed Machine Translation training scenario

   ```python
   from lunas import *
   
   X = TextLine('train.fr').select(lambda x:x.split())
   Y = TextLine('train.en').select(lambda x:x.split())
   # Each worker holds a mutually different shard of the original dataset.
   # This should be done before shuffling to reduce unnecessary shulffing efforts in each workers.
   ds = Zip(X, Y).shard(dist_word_size,dist_local_rank)
   # Construct a sample from the dataset.
   ds = ds.select(lambda x, y: 
           {
               'x': vocab_s.lookup(x),
               'y': vocab_t.lookup(y),
               'size_x': len(x),
               'size_y': len(y),
           }
       )
   ds = ds.shuffle(100000)
   # Repeat endlessly to ensure different workers can receive the same number of batches to sync.
   ds = ds.repeat() 
   
   it = SimpleIterator(ds, batch_size=128, drop_tail=False)
   
   dataloader = DataLoader(it, batch_size=4096,
         num_workers=6, 
         collate_fn=collate_fn, 
         pin_memory=True)
   
   it = iter(dataloader)
   for _ in range(max_steps):
       batch = cuda(next(it))
       ...

   ```

   - It doesn't matter if you are not familiar with machine translation task, 
   since this code should be simple enough to expain itself.

5. Continuable iteration

   ```python
   import pickle
   pickle.dump(it.state_dict(), open('state.pkl', 'wb'))
   # ...
   state = pickle.load(open('state.pkl', 'rb'))
   it.load_state_dict(state)
   
   r = Range(5)
   # outputs: [0, 1, 2]
   for i,x in enumerate(r):
       print(x)
       if i == 2:
           break
   # outputs: [3, 4]
   for i,x in enumerate(r):
       print(x)
   ```
    
   - `state_dict()` returns a picklable dictionary, 
   which can be loaded by `it.load_state_dict()` to resume the iteration process later.
   - lunas provides limited support for resumable iteration for `Dataset`, `Iterator` and `Dataloader`. 
   The iteration state is maintained by a iteration pointer in `Dataset`. 
   For those classes that manage dataset by internal buffering, 
   including `Shuffle`, `Sort`, `BucketIterator` and `DataLoader`, 
   this would NOT recover the dataset elements in the buffer.

6. Extend the dataset

   - You can refer to the implementation of `TextLine` to customize your own data dataset.

## Conclusions

Please feel free to contact if you have any question or find any bug of Lunas.
