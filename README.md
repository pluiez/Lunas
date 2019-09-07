
# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.3.5-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python based library that provides a set of simple interfaces for data processing pipelines and an iterator for looping through data.

Basically, Lunas draws its data-handling style on *Tensorflow*, *PyTorch*, and some implementation details from *AllenNLP*.

## Overview

A `Dataset` represents a dataset and holds corresponding pre-processing and filtering operations. Currently the following features are supported:

1. Buffered reading.
2. Buffered shuffling.
3. Chained processing and filtering interface.
4. Handling multiple input sources.
5. Persistable.

Supported datasets:

1. `Zip`: Zips multiple datasets.
2. `Shuffle`: A wrapper that performs buffered shuffling.
3. `Sort`: A wrapper that performs buffered sorting.
4. `InvertibleSort`: A wrapper that performs buffered sorting., and returns the sample along with its original index in the dataset.
5. `Enumerate`: Similar to Python's `enumerate` that wraps a dataset and attach an index to each element of it.
6. `Range`: Similar to Python's `range`.
7. `Count`: Similar to Python's `itertools.count`.
8. `TextLine`: A wrapper that wraps a plain-text file. Each line of the file is taken as a sample of the dataset.
9. `Stdin`: A wrapper that reads from standard input.



An `Iterator` generates batches by iterating through the dataset and maintains the iteration state. The following features are supported:

1. Dynamic batching at runtime.
2. Custom stopping criteria.
3. Persistable.

We also modify PyTorch's DataLoader to make it compatitble with our batch iterator.

*Persistable* provides the class with a *PyTorch* compatible interface to dump and load instance state, useful to resume the training process.

## Requirements

- Numpy
- overrides
- typings
- Python >= 3.7

Lunas hardly relies on any third-party libraries, all the required libraries are just
to take advantage of the type hint features provided by Python 3.

Type hint feature is used in this project and the built-in typing module of Python version lower than 3.7 can decrease the performance. However, this is solved since Python 3.7. So Lunas currently requires Python 3.7 to work efficiently.

## Installation

Install using pip:

```
pip install lunas
```

## Example

1. Create a dataset and iterate through it.

   ```python
   from lunas import Range

   ds = Range(10)
   for x in ds: # epoch 1
       print(x)
   for x in ds: # epoch 2
       print(x)
   ```

   - A `Range` dataset is created similar to range(10) and iterate through it for one epoch.
   As you see, we can iterate through this dataset several times.

2. Build a data processing pipeline.

   ```python
   ds = Range(10).select(lambda x: x + 1).select(lambda x: x * 2).where(lambda x: x % 2 == 0)
   ```

   - The chaining calls of a `Dataset` obbject defines a processing pipeline on the dataset.
   - `select(fn)` applys transformations on a dataset element lazily. The argument `fn` is a custom mapping fucntion that takes a single sample as input and output. You can apply any transformations to the dataset and return a sample of any type, e.g., `Dict`, `List` and a custom `Sample`.
   - `where(fn)` accepts a predicate and returns a `bool` value to filter an input sample, if True, the sample is preserved, otherwise discarded.
   - The mapping and filtering ops given by `select(fn)` and `where(fn)` are not executed immediately, but later when iterating through the dataset object.
   - Both `select(fn)` and `where(fn)` returns the dataset itself just to enable chaining style invocations. The mapping and filtering ops are attched to the dataset in an in-place fasion.

3. Deal with multiple input sources.

   ```python
   from lunas import Range, Zip, Shuffle

   ds1 = Range(10)
   ds2 = Range(start=10, stop=20, step=1)
   ds = Zip(ds1, ds2).select(lambda x,y: x + y)
   ds = Shuffle(ds, bufsize=5)
   ```

   - In the above code, we create two datasets and *zip* them as a `Zip` dataset. A `Zip` dataset returns a tuple from its internal datasets.
   - `Shuffle` performs randomized shuffling on the dataset.

4. Practical use case in Machine Translation scenario.

   ```python
   from lunas import TextLine, Zip, Shuffle, Sort, Iterator

   # Tokenize the input into a list of tokens.
   source = TextLine('train.fr').select(lambda x: x.split())
   target = TextLine('train.en').select(lambda x: x.split()) 
   # Ensure the inputs are of length no exceeding 50.
   ds = Zip(source, target).select(lambda x, y: 
		   {
			   x: src_vocab.lookup(x), # Map words to ids
			   y: tgt_vocab.lookup(y),
			   size_x: len(x),
			   size_y: len(y),
		   }
	   )
   ds = ds.where(lambda x: max(x['size_x'], x['size_y']) <= 50)
   # Shuffle the dataset within a buffer with bufsize 100000
   ds = Shuffle(ds, bufsize=10000)
   # Sort samples in batch by source text length
   sort_key = lambda x: len(x['size_x'])
   ds = Sort(ds, bufsize=1000, sort_key_fn=sort_key)

   # Convert a list of samples to model inputs
   collate_fn = lambda x: ...

   it = Iterator(dataset=ds, batch_size=4096, 
	     sample_size_fn=lambda x: x['size_x'], 
	     collate_fn=collate_fn, 
		 dist_world_size=1,
		 dist_local_rank=0,
	     drop_tail=True)

   # Iterate 100 epoch and 1000000 steps at most.
   for batch in it.while_true(lambda: it.epoch < 100 and it.step < 1e6):
       print(it.epoch, it.step, it.step_in_epoch, batch)

   ```

   - This code should be simple enough to understand, even if you are not familiar with machine translation.

5. Save and reload iteration state.

   ```python
   import pickle
   pickle.dump(it.state_dict(), open('state.pkl', 'wb'))
   # ...
   state = pickle.load(open('state.pkl', 'rb'))
   it.load_state_dict(state)
   ```

   - `state_dict()` returns a picklable dictionary, which can be loaded by `it.load_state_dict()` to resume the iteration process later.

6. Extend the dataset.

   - You can refer to the implementation of `TextLine` dataset to customize your own data dataset.

## Conclusions

Please feel free to contact me if you have any question or find any bug of Lunas.
