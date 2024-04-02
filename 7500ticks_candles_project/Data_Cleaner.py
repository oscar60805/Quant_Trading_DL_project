import warnings
import os
import time
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from pandas.core.frame import DataFrame


def raw_cleaner(indataframe: DataFrame) -> DataFrame:
    # 根據搓合戳記融合成Transaction Data
    temp: Dict[str, int] = defaultdict(int)
    for i in range(len(indataframe)):
        if not temp[indataframe['Rubbing Mark'][i]]:
            temp[indataframe['Rubbing Mark'][i]] = i
        else:
            if indataframe['Quantity'][temp[indataframe['Rubbing Mark'][i]]] < indataframe['Quantity'][i]:
                temp[indataframe['Rubbing Mark'][i]] = i
    indexes: List[int] = list(temp.values())
    cleaned_data: DataFrame = indataframe.loc[indexes, ['ID', 'Price', 'Quantity']]
    cleaned_data['Timestamp'] = pd.to_datetime(indataframe['Date'].astype(str) + ' ' + indataframe['Timestamp'],
                                               format='%Y%m%d %H:%M:%S.%f')
    return cleaned_data.sort_values('Timestamp')


def process_data(year_folder: str, target_data_dir: str) -> None:
    # 處理指定年份資料夾下的所有日資料
    for day_folder in os.listdir(year_folder):
        day_folder_path: str = os.path.join(year_folder, day_folder)
        transactionData_path: str = os.path.join(target_data_dir, os.path.basename(year_folder))
        processDayData(day_folder_path, transactionData_path)


def processDayData(day_folder_path: str, transactionData_path: str) -> None:
    # 處理單日資料
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=pd.errors.DtypeWarning)
            df: DataFrame = pd.read_csv(day_folder_path, sep='\t', header=None)
    except pd.errors.DtypeWarning:
        print(f"Warning: Skipping file due to DtypeWarning - {day_folder_path}")
        return

    raw_unclean: DataFrame = df.iloc[:, [0, 1, 11, 12, 14, 15]]
    raw_unclean.columns = ['Date', 'ID', 'Price', 'Quantity', 'Rubbing Mark', 'Timestamp']

    new: DataFrame = raw_cleaner(raw_unclean)
    grouped: pd.core.groupby.DataFrameGroupBy = new.groupby('ID')
    for name, group in grouped:
        processGroupData(name, group, transactionData_path)


def processGroupData(name: str, group: DataFrame, transactionData_path: str) -> None:
    # 處理分組數據
    cleaned_name: str = name.strip().replace("/", "_")
    if cleaned_name[:3] not in ['MXF', 'TXF']:
        return
    id_folder_path: str = os.path.join(transactionData_path, cleaned_name)
    os.makedirs(id_folder_path, exist_ok=True)

    tmp: pd.core.groupby.DataFrameGroupBy = group.groupby(pd.Grouper(key='Timestamp', freq='50S'))
    prev: DataFrame = pd.DataFrame()
    for interval, sub_group in tmp:
        trend: str = "no_trend"
        if not prev.empty and sub_group.iloc[-1]['Price'] - prev.iloc[0]['Price'] > 5:
            trend = "up_trend"
        if not prev.empty:
            prev['Trend'] = trend
            saveGroupData(prev, cleaned_name, interval, id_folder_path)
        prev = sub_group


def saveGroupData(group: DataFrame, name: str, interval, folder_path: str) -> None:
    # 保存處理後的分組數據
    filename: str = f'{name}_{interval.strftime("%Y%m%d_%H%M%S")}.csv'
    file_path: str = os.path.join(folder_path, filename)
    group.to_csv(file_path, index=False)


def years_process(path: str, tpath: str) -> None:
    years: list[str] = os.listdir(path)

    start_time: float = time.time()
    for year in years:
        year_path: str = os.path.join(path, year)
        target_year_dir: str = os.path.join(tpath, year)
        os.makedirs(target_year_dir, exist_ok=True)
        print(f"Year: {year}，Processing {year} Data Start")
        process_data(year_path, target_year_dir)
    end_time: float = time.time()
    total_time: float = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 打印結果
    print(f"Total processing time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")


def main() -> None:
    root_data_dir: str = "..."
    target_data_dir: str = "..."
    years_process(root_data_dir, target_data_dir)


if __name__ == '__main__':
    main()
