# :sunny::umbrella: THIES Binary Processor

This repository processes binary data from the [THIES Data Logger DL16](https://www.thiesclima.com/en/Products/Miscellaneous-Devices-Data-logger/?art=992). It provides functions to read binary files and export them in CSV format.

#### Data Directory

The binary data export from THIES DL16 should follow this structure (the dates and `DIRNAME` are just examples):

```bash
./DIRNAME/
    ├── ARCH_AV1/
    |   ├── 20240425.BIN
    |   ├── ...
    |   ├── 20240603.BIN
    |   └── DESCFILE.INI
    └── ARCH_EX1/
        ├── 20240425.BIN
        ├── ...
        ├── 20240603.BIN
        └── DESCFILE.INI
```

`ARCH_AV1` contains average values for each parameter.
`ARCH_EX1` contains min/max values.

## Usage

Check the `bin2txt.ipynb` notebook for usage examples.

For local use, make sure all dependencies are installed. This code was developed using Conda. It mainly uses:

- Python 3.8.8
- Pandas 1.2.4
- Numpy 1.20.1
- Bitarray 1.9.2

Dependencies are listed in `requirements.txt` file. If you're using Conda, you can run the following command:

```bash
conda install --file requirements.txt
```

### THIESDayData

This class is made to handle a single .BIN file that represents one day's data. To load a .BIN file, for example:

```python
data = THIESDayData(datatype='AV')
data.read_binfile("ARCH_AV1/20240425.BIN", "ARCH_AV1/DESCFILE.INI")
```

To export the data:

```python
data.write_csv('output_filename')
```

### THIESData

This class is made to handle multiple .BIN files and export them as a single CSV file.

First we create a THIESData instance. In this example we will read average (AV) values:

```python
data = THIESData(datatype='AV', dirpath='ARCH_AV1')
```

We can export the data in 2 different ways: using the `read_write()` method or the load-and-write method.

**1. `read_write()`:** After creating `data`, we simply run

```python
data.read_write(outpath='output_filename')
```

This will create a new `output_filename.csv` file with all the binary data stored in the `/ARCH_AV1` directory.

**2. Load and write:** We first load the data from the directory into the THIESData instance.

```python
data.load_df()
```

This will load the data as a Pandas DataFrame that can be accessed as:

```python
data.fullData.dataDF
```

To create the CSV output file:

```python
data.df2csv(outpath='output_filename')
```

#### Contributions

For bug reports, requests or other problems please sumbit an issue.

#### Contact info

If you have any questions, issues, or comments, you can contact me at:

Email: socovacich@uc.cl
Github: @sopadeoliva
