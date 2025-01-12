# Cell Linking Benchmark

`link.py` is the python script that we submitted to the Cell Linking Benchmark ([CLB](https://celltrackingchallenge.net/latest-clb-results/)).

This guide will help you reproduce our results.

## Environment creation and installation

We recommend using conda to set up the environment. Here are the few command lines to reproduce:

```bash
$ conda create -n byotrack python=3.10
$ conda activate byotrack
$ pip install -r requirements.txt  # Without conda, you can simply run this line to install the requirements on your python environment
```

## Running the linker

To run a single tracking on a specific dataset:

```bash
$ # Activate your environement if not already done
$ conda activate byotrack
$
$ # Run the script for the specific dataset.
$ # Replace $CTC_PATH by the path to the parent folder of each CTC datasets.
$ # The code expects to find a dataset with CTC format at {data_path}/{dataset}
$ # Some extra parameters can be fixed for some specific dataset (see `run_clb.py` or `hyper_parameters.json`)
$ python link.py --data_path $CTC_PATH --dataset BF-C2DL-HSC --seq 1
```

To run on all datasets of the CLB, we provided the `run_clb.py` script:

```bash
$ # Activate your environement if not already done
$ conda activate byotrack
$
$ # Run on all datasets/seq.
$ # Use --default_parameters to build a true general submission (non-dataset specific, even for hyper parameters)
$ python run_clb.py --data_path $CTC_PATH  # --default-parameters
```
