import os
import pandas as pd
import random
import time


def make_trainData(path: str, save_path: str) -> None:
    ids: list[str] = os.listdir(path)
    train_path: str = os.path.join(save_path, 'train')
    val_path: str = os.path.join(save_path, 'val')
    train_up_path: str = os.path.join(train_path, 'up_trend')
    train_no_up_path: str = os.path.join(train_path, 'no_up_trend')
    val_up_path: str = os.path.join(val_path, 'up_trend')
    val_no_up_path: str = os.path.join(val_path, 'no_up_trend')

    os.makedirs(train_up_path, exist_ok=True)
    os.makedirs(train_no_up_path, exist_ok=True)
    os.makedirs(val_up_path, exist_ok=True)
    os.makedirs(val_no_up_path, exist_ok=True)

    for id in ids:
        id_path: str = os.path.join(path, id)
        candles: list[str] = os.listdir(id_path)
        random.shuffle(candles)  # Shuffle the order
        length: int = len(candles)
        val_num: int = int(length * 0.2)
        for index, candle in enumerate(candles):
            candle_path: str = os.path.join(id_path, candle)
            temp: pd.DataFrame = pd.read_csv(candle_path)
            trend: str = temp['Trend'].iloc[0]
            selected_data: pd.DataFrame = temp[['Price', 'Quantity']]

            if index < val_num:
                file_path: str = os.path.join(val_up_path if trend == 'up_trend' else val_no_up_path, candle)
            else:
                file_path: str = os.path.join(train_up_path if trend == 'up_trend' else train_no_up_path, candle)
            selected_data.to_csv(file_path, index=False)


def years_process(path: str, save_path: str) -> None:
    years: list[str] = os.listdir(path)
    for year in years:
        year_path: str = os.path.join(path, year)
        year_save_path: str = os.path.join(save_path, year)
        start_time: float = time.time()
        print(f'{year} splitting start')

        make_trainData(year_path, year_save_path)

        end_time: float = time.time()
        elapsed_time: float = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f"{year} Splitting to Train/Validation Set took {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds to complete.")


if __name__ == '__main__':
    data_path: str = '...'
    target_path: str = '...'
    years_process(data_path, target_path)
