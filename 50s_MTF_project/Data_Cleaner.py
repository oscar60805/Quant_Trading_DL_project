import warnings
import os
from typing import Any, Dict, List
import pandas as pd
from collections import defaultdict


# 根據期交所的原始資料、對應到的相同的搓合記號: 'Rubbing Mark' 視為同筆交易
# 並且對於同筆交易取出商品名: 'ID'、時間點: 'Timestamp'、成交價格: 'Price'、成交數量: 'Quantity'
def rawCleaner(Indataframe: pd.DataFrame) -> pd.DataFrame:
    temp: Dict[Any, int] = defaultdict(int)
    for i in range(len(Indataframe)):
        if not temp[Indataframe['Rubbing Mark'][i]]:
            temp[Indataframe['Rubbing Mark'][i]] = i
        else:
            if Indataframe['Quantity'][temp[Indataframe['Rubbing Mark'][i]]] < Indataframe['Quantity'][i]:
                temp[Indataframe['Rubbing Mark'][i]] = i
    indexes: List[int] = [x for x in temp.values()]
    cleaned_data: pd.DataFrame = Indataframe.loc[indexes, ['ID', 'Price', 'Quantity']]
    cleaned_data['Timestamp'] = pd.to_datetime(Indataframe['Date'].astype(str) + ' ' + Indataframe['Timestamp'],
                                               format='%Y%m%d %H:%M:%S.%f')
    return cleaned_data.sort_values('Timestamp')


def processDayData(day_folder_path: str, transactionData_path: str) -> None:
    # 由於資料內有些微小的瑕疵(同比數據內的同column數據類型不同)、可能是紀錄資料時發生的問題，直接將整日包含瑕疵的數據排除
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=pd.errors.DtypeWarning)
            df: pd.DataFrame = pd.read_csv(day_folder_path, sep='\t', header=None)
    except pd.errors.DtypeWarning:
        # 打印出哪些是有瑕疵的數據、後須根據數據分析修改
        print(f"Warning: Skipping file due to DtypeWarning - {day_folder_path}")
        return

    raw_unclean: pd.DataFrame = df.iloc[:, [0, 1, 11, 12, 14, 15]]
    raw_unclean.columns = ['Date', 'ID', 'Price', 'Quantity', 'Rubbing Mark', 'Timestamp']

    new: pd.DataFrame = rawCleaner(raw_unclean)
    grouped: pd.core.groupby.DataFrameGroupBy = new.groupby('ID')
    for name, group in grouped:
        os.makedirs(transactionData_path, exist_ok=True)
        processGroupData(name, group, transactionData_path)


def processGroupData(name: str, group: pd.DataFrame, transactionData_path: str) -> None:
    cleaned_name: str = name.strip().replace("/", "_")
    if cleaned_name[:3] not in ['MXF', 'TXF'] or len(cleaned_name) > 5:
        return
    id_folder_path: str = os.path.join(transactionData_path, cleaned_name)
    os.makedirs(id_folder_path, exist_ok=True)
    tmp: pd.core.groupby.DataFrameGroupBy = group.groupby(pd.Grouper(key='Timestamp', freq='50S'))
    prev: pd.DataFrame = pd.DataFrame()
    for interval, sub_group in tmp:
        trend: str = "no_trend"
        if not prev.empty:
            # 辨別趨勢，這裡的趨勢規則採用分析後的5點幅度判斷是否為上升趨勢
            if sub_group.iloc[-1]['Price'] - prev.iloc[0]['Price'] > 5:
                trend = "up_trend"
            if not prev.empty:
                prev['Trend'] = trend
                saveGroupData(prev, cleaned_name, interval, id_folder_path)
            prev = sub_group.copy()
        else:
            prev = pd.DataFrame()


def saveGroupData(data: pd.DataFrame, name: str, interval: pd.Timestamp, path: str) -> None:
    prev_formatted_interval: str = interval.strftime('%Y%m%d_%H%M%S')
    filename: str = f'{name}_{prev_formatted_interval}.csv'
    file_path: str = os.path.join(path, filename)
    data.to_csv(file_path, index=False)


def years_process(path: str, tpath: str) -> None:
    years: list[str] = os.listdir(path)
    for year in years:
        year_path: str = os.path.join(path, year)
        target_path: str = os.path.join(tpath, year)
        print(f"Year : {year}，Processing {year} Data Start")
        for day_folder in os.listdir(year_path):
            day_folder_path: str = os.path.join(year_path, day_folder)
            processDayData(day_folder_path, target_path)


if __name__ == '__main__':
    root_data_dir: str = "/home/oscar/AllKindData/MXF/DATA"
    target_data_dir: str = "/home/oscar/AllKindData/TransactionS/TransactionData2_1"
    years_process(root_data_dir, target_data_dir)
