# ood-playground

Hello! :vulcan_salute:

Here you may find a plenty of useful and lots of useless code.

### Setting up the Libraries:

###### 1. Install `brain-segm` module:

```
git clone https://gitlab.com/ira-labs/projects/research/ood-playground.git
cd ood-playground
pip install -e .
```

###### 2. Install `surface-distance`:
```
git clone https://github.com/deepmind/surface-distance.git
pip install surface-distance/
```

Original repository: https://github.com/deepmind/surface-distance

To remove the `DeprecationWarning: 'np.bool' is a deprecated alias
for the builtin 'bool'.` please change the line `49`
in `surface_distance/metrics.py` to

```
    if array.dtype != bool:
```

(Might be already fixed by the time you are reading this.)

### Data downloading and preprocessing

For all datasets you need to specify the local path 
to a folder with the preprocessed dataset in the file `ood/paths.py`.

#### 1. LIDC dataset
Download the dataset
[here](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
For the benchmarking purposes you need to download only "Images (DICOM, 125GB)".
Firstly, download its `.tcia` file.

Then we suggest to use NBIA Data Retriever CLI.
We also use its open source version so that NBIA Data Retriever
cloud be installed without the super user permission.

If you prefer to use NBIA Data Retriever GUI,
feel free to skip the steps below and follow the TCIA guidelines.

1. Download the `.tcia` file and place it anywhere on your computer,
e.g., `~/TCIA_LIDC-IDRI_20200921.tcia`.

2. Clone
[NBIA Data Retriever CLI repo](https://github.com/ygidtu/NBIA_data_retriever_CLI):
`git clone git@github.com:ygidtu/NBIA_data_retriever_CLI.git`

3. Install go in your conda.
(Go is needed to build the data retriever scripts.)
`conda install -c conda-forge go`

4. Build the data receiver scripts:
(1) `cd NBIA_data_retriever_CLI`, (2) `chmod +x build.sh`, (3) `./build.sh`.
    
5. Run the downloading script:

```
./nbia_cli_linux_amd64 -i ~/TCIA_LIDC-IDRI_20200921.tcia -o <raw_data_path>
``` 


You need to specify `<raw_data_path>`, where to download the data.
You may also have a platform different from linux
so you need to choose another available downloading script.   

Finally, run our preprocessing script:

```
python scripts/preproc_lidc.py -i <raw_data_path>/LIDC-IDRI -o <preprocess_data_path>
```

`raw_data_path` should be a path to a folder `LIDC-IDRI`,
which contains folders `LIDC-IDRI-{i}`.

Now you can use `<preprocess_data_path>`
as path to LIDC dataset in `ood/paths.py`