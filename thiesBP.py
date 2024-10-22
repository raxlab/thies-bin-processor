import os
import numpy as np
from bitarray import bitarray
import pandas as pd
import configparser
import struct
from datetime import datetime, timedelta


TIME_LIST = []

start_time = datetime.strptime("00:00", "%H:%M")
for i in range(0, 24 * 60, 10):
    TIME_LIST.append((start_time + timedelta(minutes=i)).strftime("%H:%M"))

ROWS = len(TIME_LIST)


def date_range(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y/%m/%d") + timedelta(days=1)
    end = datetime.strptime(end_date, "%Y/%m/%d") - timedelta(days=1)
    return [(start + timedelta(days=i)).strftime("%Y/%m/%d") for i in range((end - start).days + 1) if start <= end]


def add_date_sep(date: str) -> str:
    '''
    Input: date as YYYYMMDD.BIN
    Returns: date as YYYY/MM/DD
    '''
    return date[:4] + '/' + date[4:6] + '/' + date[6:8]


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
        for v in section_dict:
            if v == 'name':
                continue
            section_dict[v] = int(section_dict[v])
        data_dict[int(section)] = section_dict
    return data_dict


class THIESDayData:
    # Bytes per parameter
    BPP = {'av': 5, 'ex': 9}
    # Timestamp Offset
    OFFSET = 4

    def __init__(self, datatype: str) -> None:
        d = datatype.lower().strip()
        if d not in ['av', 'ex']:
            raise ValueError(
                "Invalid datatype. Expected 'av' (average values) or 'ex' (minmax values).")

        self._bpr = -1                      # Bytes per row
        self._bpp = THIESDayData.BPP[d]     # Bytes per parameter
        self._datatype = d
        self._binfile = None
        self.descfile = {}
        self.nparameters = -1
        self.nbytes = -1
        self.nrows = -1
        self._date = ''
        self.statusDF = pd.DataFrame()
        self.dataDF = pd.DataFrame()
        self.datesDF = pd.DataFrame()

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

    def _set_descfile(self, inipath: str) -> None:
        self.descfile = read_descfile(inipath)
        self.nparameters = len(self.descfile)
        row_size = sum([self.descfile[num]['size'] for num in self.descfile])
        self._bpr = row_size + THIESDayData.OFFSET

    def read_binfile(self, binpath: str, inipath: str) -> None:
        self._set_descfile(inipath)
        with open(binpath, "rb") as bin_file:
            binfile = bin_file.read()
        self._binfile = binfile
        self.nbytes = len(self._binfile)
        self.nrows = int(self.nbytes / self._bpr)
        self._make_dataframes()

    def make_empty(self, inipath: str, date: str) -> None:
        self._set_descfile(inipath)
        dataDF = pd.DataFrame(None, index=range(
            ROWS), columns=range(self.nparameters+2))
        col_names = {0: 'Date', 1: 'Time'}
        par_names = {key+1: self.descfile[key]['name']
                     for key in self.descfile}
        col_names.update(par_names)
        dataDF = dataDF.rename(columns=col_names)
        dataDF['Time'] = TIME_LIST
        dataDF['Date'] = [date]*ROWS

        self.dataDF = dataDF
        self.statusDF = dataDF
        self.datesDF = dataDF

    def _make_dataframes(self) -> None:
        '''
        Builds data DF, status DF and, if datatype=ex, dates DF.
        '''
        byterows = [self._binfile[i*self._bpr +
                                  THIESDayData.OFFSET: (i+1)*self._bpr] for i in range(0, self.nrows)]
        data_arr = np.zeros((self.nrows, self.nparameters))
        status_arr = np.zeros((self.nrows, self.nparameters))
        time_idx = np.empty(self.nrows, dtype=object)
        date_idx = np.empty(self.nrows, dtype=object)
        dates_arr = np.empty((self.nrows, self.nparameters), dtype=object)

        for i, row in enumerate(byterows):
            # Timestamp
            ts_bytes = self._binfile[i*self._bpr:i*self._bpr + 4]
            ts = THIESDayData._bytes2datetime(ts_bytes)
            date_idx[i], time_idx[i] = ts[:-3].split()

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
                    dt = THIESDayData._bytes2datetime(
                        row[j*self._bpp + 5: j*self._bpp + 9], only_time=True)
                    dates_arr[i, j] = dt

        self.dataDF = pd.DataFrame(data_arr).rename(
            columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
        self.statusDF = pd.DataFrame(status_arr).rename(
            columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
        self.dataDF = self.dataDF.where(self.statusDF == 0.0, other=None)

        if self._datatype == 'ex':
            self.datesDF = pd.DataFrame(dates_arr).rename(
                columns={i: self.descfile[i+1]['name'] for i in range(self.nparameters)})
            self.datesDF = self.datesDF.where(self.statusDF == 0.0, other=None)
            self.datesDF.insert(0, 'Time', time_idx)
            self.datesDF.insert(0, 'Date', date_idx)

        self.dataDF.insert(0, 'Time', time_idx)
        self.dataDF.insert(0, 'Date', date_idx)
        self.statusDF.insert(0, 'Time', time_idx)
        self.statusDF.insert(0, 'Date', date_idx)

    def _generate_blank_rows(self) -> list:
        if len(self) == ROWS:
            # Nothing to fill (already full rows)
            return []

        new = []
        none_row = {col: None for col in self.dataDF.columns}
        none_row['Date'] = self.date
        current_times = self.dataDF['Time']
        for time in TIME_LIST:
            if time not in current_times.values:
                row = none_row.copy()
                # 'time' was not measured in the original data
                # fill it with None row
                row['Time'] = time
                new.append(row)
        return new

    def complete_empty(self):
        '''
        Completes DataFrames with all the timestamps of missing data
        Fills all columns with 'None' except Date and Time cols
        '''
        if len(self) == ROWS:
            return
        new_rows = self._generate_blank_rows()
        self.dataDF = self.dataDF.append(new_rows, ignore_index=True)
        self.dataDF = self.dataDF.sort_values(by='Time').reset_index(drop=True)
        self.statusDF = self.statusDF.append(new_rows, ignore_index=True)
        self.statusDF = self.statusDF.sort_values(
            by='Time').reset_index(drop=True)

        if self._datatype == 'ex':
            self.datesDF = self.datesDF.append(new_rows, ignore_index=True)
            self.datesDF = self.datesDF.sort_values(
                by='Time').reset_index(drop=True)

    def sort_by(self, cols: list):
        self.dataDF = self.dataDF.sort_values(
            by=cols, ascending=[True, True]).reset_index(drop=True)
        self.statusDF = self.statusDF.sort_values(
            by=cols, ascending=[True, True]).reset_index(drop=True)
        if len(self.datesDF):
            self.datesDF = self.datesDF.sort_values(
                by=cols, ascending=[True, True]).reset_index(drop=True)

    @property
    def date(self) -> str:
        '''
        Returns str of date of measurement
        '''
        if len(self.dataDF) and self._date == '':
            self._date = self.dataDF['Date'][0]
        return self._date

    @property
    def shape(self):
        return self.dataDF.shape

    @property
    def info(self) -> None:
        bf = self._binfile
        if bf:
            bf = bf[:8]
        print(f'''=== THIES Day Data Instance ===\n
Bytes per row (BPR): {self._bpr}
Bytes per parameter (BPP): {self._bpp}
Datatype: {self._datatype}
Binfile: {bf}...
Descfile: {self.descfile}
N parameters: {self.nparameters}
N Bytes: {self.nbytes}
Rows: {self.nrows}
Date: {self.date}
    ''')

    def write_csv(self, filename: str) -> None:
        with open(filename + '.csv', 'w') as outfile:
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
            new.statusDF = pd.concat(
                [self.statusDF, other.statusDF]).reset_index(drop=True)
            new.dataDF = pd.concat(
                [self.dataDF, other.dataDF]).reset_index(drop=True)
            if self._datatype == 'ex':
                new.datesDF = pd.concat(
                    [self.datesDF, other.datesDF]).reset_index(drop=True)
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

        self._verify_path(dirpath)
        descpath = self._path + '/DESCFILE.INI'
        self.descfile = read_descfile(descpath)

        self.daylist = []
        self.fullData = pd.DataFrame()

        self.completed = False

    def reset(self):
        self.daylist = []
        self.fullData = pd.DataFrame()
        self.completed = False

    def _verify_path(self, path: str) -> None:
        fl = sorted(os.listdir(path))
        if 'DESCFILE.INI' not in fl:
            raise FileNotFoundError('No DESCFILE.INI found in this directory.')
        self.filelist = fl[:-1]

    def load_df(self, complete_rows=False) -> pd.DataFrame:
        '''Reads folder given in DIRPATH and
        transforms data into DF. Saves it in self.fullData
        - complete_rows (bool): if True, completes DFs with Empty Rows by calling
            THIESDayData.complete_empty()
        '''
        self.reset()
        for f in self.filelist:
            filepath = f'{self._path}/{f}'
            daydata = THIESDayData(datatype=self._datatype)
            daydata.read_binfile(binpath=filepath, inipath=self.descfile)
            if complete_rows:
                daydata.complete_empty()
            self.daylist.append(daydata)

        self.fullData = sum(self.daylist, start=THIESDayData(self._datatype))
        return self.fullData

    def complete_empty_dates(self):
        if self.completed:
            return
        date_s = add_date_sep(self.filelist[0])
        date_e = add_date_sep(self.filelist[-1])
        d_range = date_range(date_s, date_e)
        for date in d_range:
            if date not in self.fullData.dataDF['Date'].values:
                # Missing day
                new = THIESDayData(self._datatype)
                new.make_empty(self.descfile, date=date)
                self.fullData += new

        self.fullData.sort_by(['Date', 'Time'])
        self.completed = True

    def df2csv(self, outpath: str) -> None:
        # if self._datatype == 'av':
        # FORMAT FOR EX FILES ???
        self.fullData.write_csv(outpath)
        print(f'Data written in: {outpath}.csv')

    def read_write(self, outpath: str):
        '''Quick version of the read-write process.
        Reads the path given and writes all BIN file data in same CSV
        Does NOT save as DF the data.
        Does NOT complete missing timestamps with empty rows.
        '''
        write_header = True
        bcount = 0
        with open(outpath+'.csv', "w") as outfile:
            for i, f in enumerate(self.filelist):
                filepath = f'{self._path}/{f}'
                daydata = THIESDayData(datatype=self._datatype)
                daydata.read_binfile(binpath=filepath, inipath=self.descfile)
                outfile.write(daydata.dataDF.to_csv(header=write_header))
                bcount += daydata.nbytes
                if i == 0:
                    write_header = False
        print(f'Data written in: {outpath}.csv')

    @property
    def shape(self):
        return self.fullData.shape

    @property
    def size(self):
        return len(self.filelist)

    def __repr__(self) -> str:
        return str(self.fullData)

    def _repr_html_(self):
        return self.fullData
