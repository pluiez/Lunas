# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.5.0-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python based library that mimics TensorFlow's `dataset` API and 
also its logics to build a data processing pipeline for arbitrary datasets.

The implementation mostly draws on TensorFlow but in a simplified and pure-Python fashion. 

## Features 

A `Dataset` represents a dataset and optionally holds specific custom operations on dataset elements. 

The evaluation of operations are performed lazily, hence it's a trade-off for memory against speed.

### Datasets

Currently the following datasets are supported:

1. `TextLine`: iterates through a text file in read mode line by line.
2. `Stdin`: wraps the standard input as a dataset.
3. `Array`: wraps an iterable object as a dataset.
4. `Range`: wraps a range of integers as a dataset, simulating builtin `range`.
5. `Enumerate`: wraps a dataset with index for each element, simulating builtin `enumerate`.
6. `Zip`: wraps multiple datasets as one dataset and supports custom padding for varying-sized datasets.
7. `Concat`: concatenates multiple datasets as one dataset.
8. `Glob`: wraps the standard `glob.glob` as a dataset.
9. `Map`: transforms elements by a given mapping function.
10. `Where`: filters elements by a given predicate function.
11. `Repeat`: repeats the dataset for multiple epochs.
12. `Interleave`: maps a dataset into multiple datasets and interleave between the datasets.
13. `Shuffle`: shuffles a dataset using a buffer for memory-efficient randomisation.
14. `Sort`: sorts the dataset.
15. `Slice`: slices the dataset.
16. `Shard`: shards the dataset into different partitions.
17. `Window`: iterates through the dataset using a sliding window. 

Additionally, a chaining style dataset creation is available for 
`Map`, `Where`, `Repeat`, `Shard`, `Shuffle`, `Sort`, `Slice`, and `Window`.

For example, any dataset can invoke the following to create a dataset: 

```python
ds = Range(100).map(lambda x: 2 * x).where(lambda x: x < 50).take(10)
```

### Batch Iterators

The batch iterators are provided to yield batches from a given dataset, including: 

1. `ConstantIterator`: yields a constant number of samples for each batch.
2. `BucketIterator`: yields varying-sized batch, in which the size of each sample is determined by a given function. 
3. `DataLoader`: wraps PyTorch's `torch.utils.data.DataLoader` to provide multiprocessing data-loading features.

### Persistence

Both datasets and batch iterators support persistence using `state()` and `load()` interface. `state()` takes a snapshot of the current iteration state into a dictionary, while `load()` restores iteration from a specific state later. 

## Requirements

- Python >= 3.7
- numpy
- pytorch >= 1.5.0

## Installation

Install using pip:

```shell
pip install lunas
```

## Basics

1. Create a dataset and iterate through it:

   ```python
   from lunas import Range
   
   ds = Range(1000).shuffle(buffer_size=100)
   for x in ds: # epoch 1
       print(x)
   for x in ds: # epoch 2
       print(x)
   
   ds = Range(1000).shuffle(buffer_size=100).repeat(2)
   for x in ds: # 2 epochs
       print(x)
   ```

    - A dataset can be scanned through for several epochs.
    - Dataset.shuffle() performs a buffered shuffling. The shuffling does not happens at dataset creation, but rather begins when trying to access an element from the dataset. 
    - Alternatively, `Dataset.repeat(2)` creates another dataset that 
      iterates through the original dataset twice.

2. Build a data processing pipeline:

   ```python
   from lunas import *
   ds = Range(10).map(lambda x: x * 2).where(lambda x: x % 2 == 0)
   ```

   - The chaining calls of a `Dataset` object defines a processing pipeline on the original dataset.

3. Deal with multiple data sources:

   ```python
   from lunas import *
   
   ds1 = Range(10)
   ds2 = Range(start=10, stop=20, step=1)
   ds = Zip([ds1, ds2]).map(lambda x, y: (x + y), unpack_args=True)
   
   ds3 = Range(10)
   ds4 = Range(100)
   ds5 = Range(1000)
   ds = Zip([ds3, ds4, ds5], mode='>', padding=True).map(lambda x, y, z: (x + y + z), unpack_args=True)
   ```

   - Two datasets here are zipped as a `Zip` dataset. A `Zip` dataset returns a tuple from the internal child-datasets, that is `ds1` and `ds2`.

   - `Zip` requires strictly the datasets to be aligned by default. It also allows zipping multiple datasets of different sizes by providing additional `mode` and `paddinng` argument to indicate either padding smaller dataset or truncating bigger dataset.

4. Example usage in a more complicated distributed Machine Translation training scenario:

   ```python
   from lunas import *
   
   # tokenises source langauge
   X = TextLine('train.fr').map(lambda x: x.split())
   # tokenises target language
   Y = TextLine('train.en').map(lambda x: x.split())
   # Each worker holds a mutually different shard of the original dataset
   # This should be done before shuffling to avoid unnecessary shuffling efforts in each workers.
   ds = Zip(X, Y).shard(dist_word_size, dist_local_rank)
   # Constructs a sample from the dataset
   ds = ds.map(lambda x, y: {
               'x': vocab_s.lookup(x), # convert token list into word indices
               'y': vocab_t.lookup(y),
               'size_x': len(x), # number of tokens in source language
               'size_y': len(y), # number of tokens in target language
           }, unpack_args=True
       )
   ds = ds.shuffle(100000)
   # Repeats endlessly
   ds = ds.repeat()
   
   batch_itr = BucketIterator(
       ds, 
       # each batch size is at most 4096
       batch_size=4096, 
       # size for each sample is measured in number of tokens in target language
       get_length_fn=lambda sample: sample['size_y'],   
       bucket_boundaries=get_bucket_boundaries()
   )
   
   dataloader = DataLoader(
       batch_itr, 
       batch_size=4096,
       num_workers=6, 
       collate_fn=collate_fn,
   )
   
   it = iter(dataloader)
   for _ in range(max_steps):
       batch = cuda(next(it))
       ...
   
   ```

   - It doesn't matter if you are not familiar with machine translation task, 
     since this code should be simple enough to explain itself.

5. Resume iteration:

   ```python
   import pickle
   # Stops at the 10-th element
   for i, x in enumerate(it):
       if i == 10:
           break
   pickle.dump(it.state(), open('state.pkl', 'wb'))
   # ...
   state = pickle.load(open('state.pkl', 'rb'))
   it.load(state)
   # Starts from the 11-th element
   for i, x in enumerate(it):
       ...
   ```

   - `it` here can be a dataset or batch iterator object.
   - `state()` returns a picklable dictionary, which can be loaded by `it.load()` to resume the iteration procedure later.
   - lunas provides limited support for resumable iteration. Specifically, the iteration state is maintained by a counting pointer in `Dataset`. For those dataset implementations that manage iteration by internal buffering, such as `Shuffle`, `Sort`, `BucketIterator` and `DataLoader`, `load()` would NOT recover the dataset elements in the buffer.

6. Extend the dataset:

   - You can refer to the implementation of `TextLine` to customize your own data dataset.

## Known issues

1. Parallel processing is not yet supported due to Python's limited support for parallelization. 

   Multi-threading can be helpful for resource-intensive data loading operations, but not for CPU-intensive data processing operations. Whereas multi-processing is facilitates CPU-intensive scenarios, there are a few limitations, which further introduce complexity in the use of the library. 

   Although it won't cause any difference for lunas APIs, the users will have to pay more attention in order to ensure multi-processing work correctly. For example, multi-processing does not accept lambda expressions and any unpicklable objects as arguments. The more severe problem is that once the child-process terminated with certain fatal errors (for example, a segment fault),  the parent process will never be notified the termination of the child. It thus requires extra efforts on accounting the states of child processes and 
   the standard `multiprocessing` library does not come to use. 

   We are likely to opt to C++ based implementation for parallelization features just as TensorFlow did.

2. Stdin dataset cannot be used in potential multiprocessing context.

   multiprocessing can mess up standard input since we can't distribute /dev/stdin to multiple processes with trivial implementation. Furthermore, there seems to be little preferential needs to spread stdin to multiple processes, so the problem is simply left aside.

## License

[MIT License](https://github.com/pluiez/Lunas/LICENSE)
