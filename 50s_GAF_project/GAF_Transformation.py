from typing import Tuple
from pathlib import Path
import os
import pandas as pd
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

"""
將分組好排序好的Transaction data csv文件(內部框架為['ID:期貨商品名, 'Price':成交價格, 'Quantity':成交量, 'Timestamp':時間戳記])，
再經過GAF轉換前，需先將數值投到[-1,1]區間，
這裡建構最大值最小值轉換，
由於最大值與最小值可能會有相同的情形，分母會除以0得到NaN，
故輸出前須先將NaN值轉換成 0。
"""


def min_max_normalize(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    max_price: float = data['Price'].max()
    min_price: float = data['Price'].min()
    max_quantity: float = data['Quantity'].max()
    min_quantity: float = data['Quantity'].min()
    norm_price: pd.Series = 2 * ((data['Price'] - min_price) / (max_price - min_price)) - 1
    norm_quantity: pd.Series = 2 * ((data['Quantity'] - min_quantity) / (max_quantity - min_quantity)) - 1
    return norm_price.fillna(0).to_numpy(), norm_quantity.fillna(0).to_numpy()


"""
將標準化後的 Transaction data 經過 GAF 轉換生成圖像，
並根據對應的 Label 儲存到對應的資料夾，
folder_path 為 Transaction data 存放路徑，
outer_path 為 圖像存放路徑。
"""


def save_combined_gaf_image(price_data: np.ndarray, quantity_data: np.ndarray, file_path: Path) -> None:
    image_size: int = len(price_data)
    gaf: GramianAngularField = GramianAngularField(image_size=image_size, method='summation')

    # 產生價格和數量的GAF圖像
    gaf_price: np.ndarray = gaf.fit_transform(price_data.reshape(1, -1))
    gaf_quantity: np.ndarray = gaf.fit_transform(quantity_data.reshape(1, -1))

    # 將GAF影像規範化到[0, 1]
    scaled_gaf_price: np.ndarray = (gaf_price.squeeze() + 1) / 2
    scaled_gaf_quantity: np.ndarray = (gaf_quantity.squeeze() + 1) / 2

    # 建立RGB影像：價格資訊在紅色頻道，數量資訊在綠色頻道
    rgb_image: np.ndarray = np.zeros((image_size, image_size, 3))  # 用零初始化
    rgb_image[..., 0] = scaled_gaf_price  # 红色通道
    rgb_image[..., 1] = scaled_gaf_quantity  # 绿色通道

    plt.imshow(rgb_image)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_data(transaction_path: Path, output_path: Path) -> None:
    labels: Tuple[str, ...] = ('up_trend', 'no_up_trend')
    # 為每個標籤建立目錄
    for label in labels:
        label_dir: Path = output_path / label
        label_dir.mkdir(parents=True, exist_ok=True)

    for ticker_folder in transaction_path.iterdir():
        if ticker_folder.is_dir():
            print(f'Processing {ticker_folder.name}')
            for file in ticker_folder.iterdir():
                data: pd.DataFrame = pd.read_csv(file)
                label: str = data['Trend'].iloc[0]
                norm_price, norm_quantity = min_max_normalize(data)
                if len(norm_price) > 1 and not np.all(norm_price == 0) and not np.all(norm_quantity == 0):
                    image_path: Path = output_path / label / f'{file.stem}.png'
                    save_combined_gaf_image(norm_price, norm_quantity, image_path)


def split_data_set(data_path: Path, train_folder: Path, val_folder: Path, test_folder: Path,
                   train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    # 確保訓練集、驗證集和測試集資料夾存在
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    categories: Tuple[str, ...] = ('up_trend', 'no_up_trend')

    for category in categories:

        # 建立每個類別的資料夾
        os.makedirs(os.path.join(train_folder, category), exist_ok=True)
        os.makedirs(os.path.join(val_folder, category), exist_ok=True)
        os.makedirs(os.path.join(test_folder, category), exist_ok=True)

        category_path: Path = data_path / category
        images: list[Path] = list(category_path.iterdir())
        random.shuffle(images)
        num_train: int = int(len(images) * train_ratio)
        num_val: int = int(len(images) * val_ratio)

        for i, img in enumerate(images):
            if i < num_train:
                dest_path: Path = train_folder / category
            elif i < num_train + num_val:
                dest_path: Path = val_folder / category
            else:
                dest_path: Path = test_folder / category
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dest_path / img.name)


def years_process(path: Path, tpath: Path, trainerData: Path) -> None:
    years: list[str] = os.listdir(path)
    for year in years:
        print(f'Number : {year} Transaction Data transforming Start')
        year_path: Path = path / year
        out_path: Path = tpath / year
        out_path.mkdir(parents=True, exist_ok=True)
        year_trainer_data_path: Path = trainerData / year
        train_path: Path = year_trainer_data_path / 'train_img'
        val_path: Path = year_trainer_data_path / 'val_img'
        test_path: Path = year_trainer_data_path / 'test_img'
        process_data(year_path, out_path)
        split_data_set(out_path, train_path, val_path, test_path)
        print(f'Number : {year} Transaction Data transforming End')


def main() -> None:
    transaction_all_path: Path = Path("...")
    img_all_path: Path = Path("...")
    trainerData: Path = Path("...")

    years_process(transaction_all_path, img_all_path, trainerData)


if __name__ == "__main__":
    main()
