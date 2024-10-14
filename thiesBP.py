import os
import numpy as np
from bitarray import bitarray
import pandas as pd
import configparser
import struct


def read_descfile(path) -> dict:
    ''' 
    Input: path DESCFILE.INI
    Returns: dict 
        key is index [i]
        value is dict with parameters from .ini
    '''
    if type(path) == dict:
        return path
    config = configparser.ConfigParser()
    config.read(path)
    data_dict = {}
    for section in config.sections():
        section_dict = dict(config.items(section))
        data_dict[int(section)] = section_dict
    return data_dict


def format_dt(datetime: str, date: bool, time: bool) -> str:
    '''
    Input: str "YYYY/MM/DD hh:mm:ss"
    Returns: str
    - YYYY/MM/DD (True, False), or
    - hh:mm (False, True), or
    - YYYY/MM/DD hh:mm (True, True), or
    - empty str (False, False)
    '''
    datelist = datetime.split()
    return datelist[0]*date + ' '*date*time + datelist[1][:5]*time


class THIESDayData:
    # Bytes per row
    BPR = {'av': 99, 'ex': 292}
    # Parameters per row
    PPR = {'av': 19, 'ex': 32}
    # Bytes per parameter
    BPP = {'av': 5, 'ex': 9}
    # Timestamp Offset
    OFFSET = 4

    def __init__(self, datatype: str) -> None:
        d = datatype.lower().strip()
        if d not in ['av', 'ex']:
            raise ValueError(
                "Invalid datatype. Expected 'av' (average values) or 'ex' (minmax values).")

        self._bpr = THIESDayData.BPR[d]
        self._bpp = THIESDayData.BPP[d]
        self._datatype = d
        self._binfile = None
        self.descfile = {}
        self.nparameters = -1
        self.nbytes = -1
        self.nrows = -1
        self.statusDF = pd.DataFrame()
        self.dataDF = pd.DataFrame()
        self.datesDF = pd.DataFrame()

        self._date = ''

    @staticmethod
    def _bytes2datetime(b: bytes, only_time: bool = False) -> str:
        '''
        Input: bytes (size 4)
        Output: str (YYYY/MM/DD hh:mm:ss)
        '''
        bits = bitarray()
        bits.frombytes(b[::-1])  # Invert 4 bytes
        hr = int(bits[15:20].to01(), 2)
        min = int(bits[20:26].to01(), 2)
        sec = int(bits[26:].to01(), 2)
        time = f'{str(hr).zfill(2)}:{str(min).zfill(2)}'
        if only_time:
            return time
        yr = int(bits[0:6].to01(), 2)
        mon = int(bits[6:10].to01(), 2)
        day = int(bits[10:15].to01(), 2)
        date = f'20{yr}/{str(mon).zfill(2)}/{str(day).zfill(2)}'
        return date + ' ' + time + f':{str(sec).zfill(2)}'

    def read_binfile(self, binpath: str, inipath: str) -> None:
        with open(binpath, "rb") as bin_file:
            binfile = bin_file.read()
        self._binfile = binfile
        self.nbytes = len(self._binfile)
        self.nrows = int(self.nbytes / self._bpr)
        self.descfile = read_descfile(inipath)
        self.nparameters = len(self.descfile)
        self._make_dataframes()

    def _make_dataframes(self) -> None:
        '''
        Builds data DF, status DF and, if datatype=ex, dates DF.
        '''
        byterows = [self._binfile[i*self._bpr +
                                  THIESDayData.OFFSET: (i+1)*self._bpr] for i in range(0, self.nrows)]
        data_arr = np.zeros((self.nrows, self.nparameters))
        status_arr = np.zeros((self.nrows, self.nparameters))
        timestamp_arr = np.empty(self.nrows, dtype=object)
        dates_arr = np.empty((self.nrows, self.nparameters), dtype=object)

        for i, row in enumerate(byterows):
            # Timestamp
            ts_bytes = self._binfile[i*self._bpr:i*self._bpr + 4]
            ts = THIESDayData._bytes2datetime(ts_bytes)
            timestamp_arr[i] = ts

            for j in range(self.nparameters):
                # Status = byte 1
                status = row[j*self._bpp]
                status_arr[i, j] = status

                # Value = bytes 2-5, float
                value = struct.unpack(
                    '<f', row[j*self._bpp+1: j*self._bpp+5])[0]
                data_arr[i, j] = round(value, 1)

                if self._datatype == 'ex':
                    # Datetime = bytes 6-9
                    datetime = THIESDayData._bytes2datetime(
                        row[j*self._bpp + 5: j*self._bpp + 9], only_time=True)
                    dates_arr[i, j] = datetime

        self.dataDF = pd.DataFrame(data_arr).rename(
            columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
        self.statusDF = pd.DataFrame(status_arr).rename(
            columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
        self.dataDF = self.dataDF.where(self.statusDF == 0.0, other=None)

        if self._datatype == 'ex':
            self.datesDF = pd.DataFrame(dates_arr).rename(
                columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
            self.datesDF = self.datesDF.where(self.statusDF == 0.0, other=None)
            self.datesDF.index = timestamp_arr

        # Add timestamps as index
        self.dataDF.index = timestamp_arr
        self.statusDF.index = timestamp_arr

    @property
    def date(self) -> str:
        ''' 
        Returns str of date of measurement
        '''
        if len(self.dataDF) and self._date == '':
            self._date = self.dataDF.index[0].split(' ')[0]
        return self._date

    @property
    def shape(self):
        return self.dataDF.shape

    def write_csv(self, filename: str) -> None:
        with open(filename, 'w') as outfile:
            outfile.write(self.dataDF.to_csv())

    def __repr__(self) -> str:
        return str(self.dataDF)

    def _repr_html_(self):
        return self.dataDF._repr_html_()

    def __len__(self):
        return len(self.dataDF)

    def __add__(self, other):
        if isinstance(other, THIESDayData):
            new = THIESDayData(datatype=self._datatype)
            new.descfile = self.descfile
            new.nparameters = len(new.descfile)
            new.nrows = self.nrows + other.nrows
            new.nbytes = self.nbytes + other.nbytes
            new.statusDF = pd.concat([self.statusDF, other.statusDF])
            new.dataDF = pd.concat([self.dataDF, other.dataDF])
            if self._datatype == 'ex':
                new.datesDF = pd.concat([self.datesDF, other.datesDF])
            return new
        raise TypeError(
            f"unsupported operand type(s) for +: 'THIESDayData' and '{type(other)}'")


class THIESData:

    def __init__(self, datatype: str, dirpath: str) -> None:
        d = datatype.lower().strip()
        if d not in ['av', 'ex']:
            raise ValueError(
                "Invalid datatype. Expected 'av' (average values) or 'ex' (minmax values).")

        self._path = dirpath
        self._datatype = d
        self.filelist = []
        self.size = 0
        self.descfile = {}

        self.verify_path(dirpath)

        self.daylist = []
        self.fullData = pd.DataFrame()

        self.read_folder()

    def verify_path(self, path: str) -> None:
        fl = sorted(os.listdir(path))
        if 'DESCFILE.INI' not in fl:
            raise FileNotFoundError('No DESCFILE.INI found in this directory.')
        self.filelist = fl[:-1]
        self.size = len(self.filelist)

    def read_folder(self):
        descpath = self._path + '/DESCFILE.INI'
        self.descfile = read_descfile(descpath)
        for f in self.filelist:
            filepath = f'{self._path}/{f}'
            daydata = THIESDayData(datatype=self._datatype)
            daydata.read_binfile(binpath=filepath, inipath=self.descfile)
            self.daylist.append(daydata)

        self.fullData = sum(self.daylist, start=THIESDayData(self._datatype))
        return self.fullData

    def write_csv(self, filename: str) -> None:
        # if self._datatype == 'av':
        self.fullData.write_csv(filename + '.csv')
        print(f'Done! Data written in: {filename}.csv')

    def __repr__(self) -> str:
        return str(self.fullData)

    def _repr_html_(self):
        return self.fullData
