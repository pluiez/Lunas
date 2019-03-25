
# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.3.2-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python 3-based library that provides a set of simple interfaces for data processing pipelines and an iterator for looping through data.

Basically, Lunas draws its data-handling style on *Tensorflow*, *PyTorch*, and some implementation details from *AllenNLP*.

## Overview

`Reader` defines a dataset and corresponding preprocessing and filtering pipelines. Currently the following features are supported:

1. Buffered reading.
2. Buffered shuffling.
3. Chained processing and filtering interface.
4. Preprocess and filter the data buffer in parallel.
5. Handling multiple input sources.
6. Persistable.

`Iterator` performs arbitrary iterations over the dataset and maintains the iteration state:

1. Dynamic batching at runtime.
2. Custom stopping criteria.
3. Sort samples of a batch, which is useful for learning text presentation by RNNs in *PyTorch*.
4. Persistable.

`GroupIterator` yields multiple batches at a time, 
useful for simulated large batch training on limited computational resources.

`Distributed` splits a reader for distributed training.

*Persistable* provides the class with a *PyTorch* like interface to dump and load instance state, useful when the training process is accidentally aborted.

## Requirements

- Numpy
- overrides
- typings
- Python >= 3.7

Lunas hardly relies on any third-party libraries, all the required libraries are just
to take advantage of the type hint feature provided by Python 3.

Type hint feature is used in this project and the built-in typing module for Python version lower than 3.7 can decrease the performance. However, this is solved since Python 3.7. So Lunas currently requires python 3.7 to work efficiently.

## Installation

You can simply install Lunas by running pip:

```
pip install lunas
```

## Example

*Lunas* exposes minimal interfaces to the user so as to make it as simple as possible. We try to avoid adding any unnecessary features to keep it light-weight.

However, you can still extend this library to suit your needs at any time to handle arbitrary data types such as text, images, and audios.

1. Create a dataset reader and iterate through it.

   ```python
   from lunas import Range

   ds = Range(10)
   for sample in ds:
       print(sample)
   for sample in ds:
       print(sample)
   ```

   - We create a dataset similar to range(10) and iterate through it for one epoch.
   As you see, we can iterate through this dataset several times.

2. Build a data processing pipeline.

   ```python
   ds = Range(10).select(lambda x: x + 1).select(lambda x: x * 2).where(lambda x: x % 2 == 0)
   ```

   - we call `Reader.select(fn)` to define a processing procedure for the dataset.
   - `select()` returns the dataset itself to enable chaining invocations. You can apply any transformations to the dataset and return a sample of any type, say `Dict`, `List` and custom `Sample`.
   - `where()` accepts a predicate and returns a `bool` value to filter input sample, if True, the sample is preserved, otherwise discarded.
   - It should be noted that the processing is not executed immediately, but will be performed when iterating through `ds`.

3. Deal with multiple input sources.

   ```python
   from lunas import Range, Zip, Shuffle

   ds1 = Range(10)
   ds2 = Range(10)
   ds = Zip(ds1, ds2).select(lambda x,y: x + y)
   ds = Shuffle(ds)
   ```

   - In the above code, we create two datasets and *zip* them as a `Zip` reader. A `Zip` reader returns a tuple from its internal `readers`.
   - `Shuffle` performs randomized shuffling on the dataset.

4. Practical use case in Machine Translation scenario.

   ```python
   from lunas import TextLine, Iterator

   # Tokenize the input into a list of tokens.
   source = TextLine('train.fr').select(lambda x: x.split())
   target = TextLine('train.en').select(lambda x: x.split()) 
   # Ensure the inputs are of length no exceeding 50.
   ds = Zip(source, target).select(lambda x, y: 
		   {
			   x: src_vocab.convert(x),
			   y: trg_vocab.covert(y),
			   size_x: len(x),
			   size_y: len(y),
		   }
	   )
   ds = ds.where(lambda x: max(x['size_x'], x['size_y']) <= 50)
   # Map word to id.
   ds = Shuffle(ds, shufsize=-1)

   # Convert a list of samples to model inputs
   collate_fn = lambda x: ...
   # Sort samples in batch by source text length
   sort_key = lambda x: len(x['size_x'])

   it = Iterator(ds, batch_size=4096, 
	     cache_size=4096 * 32, 
	     sample_size_fn=lambda x: x['size_x'], 
	     collate_fn=collate_fn, 
	     sort_cache_by=sort_key)

   # Iterate 100 epoch and 1000000 steps at most.
   for batch in it.while_true(lambda: it.epoch < 100 and it.step < 1e6):
   	print(it.epoch, it.step, it.step_in_epoch, batch)

   ```

   - This code should be simple enough to understand, even if you are not familiar with machine translation.

2. Save and reload iteration state.

   ```python
   import pickle
   pickle.dump(it.state_dict(), open('state.pkl', 'wb'))
   # ...
   state = pickle.load(open('state.pkl', 'rb'))
   it.load_state_dict(state)
   ```

   - `state_dict()` returns a picklable dictionary, which can be loaded by `it.load_state_dict()` to resume the iteration process later.

3. Extend the reader.

   - You can refer to the implementation of `Text` reader to customize your own data reader.

## Conclusions

Please feel free to contact me if you have any question or find any bug of Lunas.