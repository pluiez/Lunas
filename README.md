# Lunas

[![PyPI version](https://img.shields.io/badge/pypi-v0.5.0-limegreen.svg)](https://github.com/pluiez/lunas)

**Lunas** is a Python based library that mimics TensorFlow's `dataset` API and also its logics to build a data
processing pipeline for arbitrary datasets.

The implementation mostly draws on TensorFlow but in a simplified and pure-Python fashion.

## License

This project uses [MIT](LICENSE) license.

## Features

A `Dataset` represents a dataset and optionally holds custom operations on dataset elements.

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
8. `Group`: group several samples together.
9. `Flatten`: flattens a sample into multiple samples.
10. `Glob`: wraps the standard `glob.glob` as a dataset.
11. `Map`: transforms elements by a given mapping function.
12. `Where`: filters elements by a given predicate function.
13. `Repeat`: repeats the dataset for multiple epochs.
14. `Interleave`: maps a dataset into multiple datasets and interleave between the datasets.
15. `Shuffle`: shuffles a dataset using a buffer for memory-efficient randomisation.
16. `Sort`: sorts the dataset.
17. `Slice`: slices the dataset.
18. `Shard`: shards the dataset into different partitions.
19. `Sampling`: draws samples from several datasets given a sampling distribution.

Additionally, chaining-style dataset operation is available for following datasets:
`Map`, `Where`, `Repeat`, `Shard`, `Shuffle`, `Sort`, `Slice`, `Enumerate`, `Group`, `Flatten` and `Concat`.

For example, a dataset can invoke the following to create a new dataset:

```python
ds = lunas.Range(100)
.map(lambda x: 2 * x)
.where(lambda x: x < 50)
.shuffle(buffer_size=100)

print(list(ds))
```

### Batch Iterators

The batch iterators are provided to generate batches from a given dataset, currently including:

1. `ConstantIterator`: generates batches with a constant number of samples.
2. `BucketIterator`: generates varying-sized batches with sample size determined by a custom function.
3. `DataLoader`: wraps PyTorch's `torch.utils.data.DataLoader` to provide multiprocessing data-loading features.

### Persistence

Both datasets and batch iterators support persistence using `state()` and `load()` interface.
`state()` takes a checkpoint of current iteration state, while `load()` restores iteration state from a given
checkpoint.

## Requirements

- Python >= 3.7
- numpy
- pytorch >= 1.5.0

## Installation

Install using pip:

```shell
pip install -U lunas
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
    - Dataset.shuffle() performs a buffered shuffling. The shuffling does not happen immediately at dataset creation,
      but rather begins when trying to access an element from the dataset.
    - Alternatively, `Dataset.repeat(2)` creates another dataset that iterates through the original dataset twice.

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

    - Two datasets here are zipped as a `Zip` dataset. A `Zip` dataset returns a tuple from the internal child-datasets,
      that is `ds1` and `ds2`.

    - `Zip` requires strictly the datasets to be aligned by default. It also allows zipping multiple datasets of
      different sizes by providing additional `mode` and `paddinng` argument to indicate either padding smaller dataset
      or truncating bigger dataset.

4. Example usage in a more complicated distributed multilingual Language Modeling training case:

   ```python
   from lunas import *
   
   
   corpus_paths = ['train.zh', 'train.en', 'train.ru']
   sampling_weights = [0.3, 0.4, 0.3]
      
   # Shards a dataset so that each worker holds a unique shard of the original corpus.
   # Sharding should be done before shuffling to avoid unnecessary shuffling efforts in each worker.
   datasets = []
   for corpus in corpus_paths:
       ds = TextLine(corpus) \
           .shard(dist_word_size, dist_local_rank) \
           .shuffle(buffer_size=10000)
       # Tokenizes plain text into token ids
       ds = ds.map(lambda x: {'input': tokenizer.tokenize(x)})
       # Group consecutive 128 samples together, then concat and split the samples in that group into the same length
       # to reduce padding. Finally, flatten the samples group into separate samples.
       ds = ds.group(group_size=128) \
           .map(lambda xs: concat_and_split(xs, target_length=1024)) \
           .flatten()
   
       datasets.append(ds)
   # Defines a sampling strategy from the datasets
   ds = Sampling(datasets, sampling_weights, virtual_size=1000000)
   
   batch_itr = BucketIterator(
       ds,
       # each batch size has at most 4096 tokens
       batch_size=4096,
       # size for each sample is measured in number of tokens in target language
       get_length_fn=lambda x: len(x),
       bucket_boundaries=get_bucket_boundaries()
   )
   
   dataloader = DataLoader(
       batch_itr,
       num_workers=6,
       collate_fn=collate_fn,
   )
   
   for epoch in range(max_epoch):
       for bathc in dataloader:
           ...
   ```

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
    - `state()` returns a picklable dictionary, which can be loaded by `it.load()` to resume the iteration.
    - lunas provides limited support for resumable iteration. Specifically, the iteration state is maintained by a
      counting pointer in `Dataset`. For those dataset implementations that manage iteration by internal buffering, such
      as `Shuffle`, `Sort` and `BucketIterator`, `load()` would loss content in the buffer.

6. Extend the dataset:

    - You can refer to the implementation of `TextLine` to customize your own data dataset.

## Known issues

1. Parallel processing is not yet supported due to Python's limited support for parallelization.

   Multi-threading can be helpful for resource-intensive data loading operations, but not for CPU-intensive data
   processing operations. Whereas multi-processing is facilitates CPU-intensive scenarios, there are a few limitations,
   which further introduce complexity in the use of the library.

   Although it won't cause any difference for lunas APIs, the users will have to pay more attention in order to ensure
   multi-processing work correctly. For example, multi-processing does not accept lambda expressions and any unpicklable
   objects as arguments. The more severe problem is that once the child-process terminated with certain fatal errors (
   for example, a segment fault), the parent process will never be notified the termination of the child. It thus
   requires extra efforts on accounting the states of child processes and the standard `multiprocessing` library does
   not come to use.

   We are likely to opt to C++ based implementation for parallelization features just as TensorFlow did.

2. Stdin dataset cannot be used in potential multiprocessing context.

   multiprocessing can mess up standard input since we can't distribute /dev/stdin to multiple processes with trivial
   implementation. Furthermore, there seems to be little preferential needs to spread stdin to multiple processes, so
   the problem is simply left aside.
