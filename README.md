# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python 3-based library that provides a set of simple interfaces for data processing pipelines and an iterator for looping through data.

Basically, Lunas draws its data-handling style on *Tensorflow*, *PyTorch*, and some implementation details from *AllenNLP*.

## Features

`Reader` A reader defines a dataset and corresponding preprocessing and filtering rules. Currently the following features are supported:

1. Buffered reading.
2. Buffered shuffling.
3. Chained processing and filtering interface.
4. Preprocess and filter the data buffer in parallel.
5. Handling multiple input sources.
6. Persistable.

`DataIterator` An iterator performs multi-pass iterations over the dataset and maintains the iteration state:

1. Dynamic batch size at runtime.
2. Custom stopping criteria.
3. Sort samples of a batch, which is useful for learning text presentation by RNNs in *PyTorch*.
4. Persistable.

*Persistable* provides the class with a *PyTorch* like interface to dump and load instance state, useful when the training process is accidentally aborted.

## Requirements

- Numpy
- overrides
- typings
- Python = 3.x

Lunas hardly relies on any third-party libraries, all the required libraries are just
to take advantage of the type hint feature provided by Python 3.

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
   from lunas.reader import RangeReader

   ds = RangeReader(10)
   for sample in ds:
       print(sample)
   ```

   - We created a dataset similar to range(10) and iterate through it for one epoch.

2. Build a data processing pipeline.

   ```python
   ds = RangeReader(10).select(lambda x: x + 1).select(lambda x: x * 2).where(lambda x: x % 2 == 0)
   ```

   - we call `Reader.select(fn)` to define a processing procedure for the dataset.
   - `select()` returns the dataset itself to enable chaining invocations. You can apply any transformations to the dataset and return a sample of any type, say `Dict`, `List` and custom `Sample`.
   - `where()` accepts a predicate and returns a `bool` value to filter input sample, if True, the sample is preserved, otherwise discarded.
   - It should be noted that the processing will not be executed immediately, but will be performed when iterating through `ds`.

3. Deal with multiple input sources.

   ```python
   from lunas.reader import RangeReader, ZipReader, ShuffleReader

   ds1 = RangeReader(10)
   ds2 = RangeReader(10)
   ds = ZipReader(ds1, ds2).select(lambda x: x[0] + x[1])
   ds = ds.shuffle()
   ```

   - In the above code, we created two datasets and *zip* them as a `ZipReader`. A `ZipReader` returns a tuple from its internal `readers`.
   - `ds.shuffle()` returns a `ShuffleReader` of the dataset.

4. Practical use case in Machine Translation scenario.

   ```python
   from lunas.readers.text import TextReader
   from lunas.iterator import DataIterator

   # Tokenize the input into a list of tokens.
   tokenize = lambda line: line.split()
   # Ensure the inputs are of length no exceeding 50.
   limit = lambda src_tgt: max(map(len, src_tgt)) <= 50
   # Map word to id.
   word2id = lambda src_tgt: ...

   source = TextReader('train.fr').select(tokenize)
   target = TextReader('train.en').select(tokenize)
   ds = ZipReader(source, target).where(limit)
   ds = ds.shuffle().select(word2id)

   # Take maximum length of the sentence pair as sample_size
   sample_size = lambda x: max(map(len), x)
   # Convert a list of samples to model inputs
   collate_fn = lambda x: ...
   # Sort samples in batch by source text length
   sort_key = lambda x: len(x[0])

   it = DataIterator(ds, batch_size=4096, cache_size=40960, sample_size_fn=lambda x, collate_fn=collate_fn, sort_desc_by=sort_key)

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

   - You can refer to the implementation of `TextReader` to customize your own data reader.

## Conclusions

Please feel free to contact me if you have any question or find any bug of Lunas.