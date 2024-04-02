import os
import pandas as pd
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
from typing import List
from pyts.image import MarkovTransitionField


def generate_mtf_image(data: pd.Series, image_size: int) -> np.ndarray:
    # 使用MTF方法將序列轉換為影像。
    mtf: MarkovTransitionField = MarkovTransitionField(image_size=image_size, n_bins=5, strategy='uniform')
    mtf_matrix: np.ndarray = mtf.fit_transform(data.values.reshape(1, -1)).squeeze()
    return mtf_matrix


def save_image(image: np.ndarray, file_path: str) -> None:
    # 儲存影像到指定路徑。
    plt.imshow(image, cmap='viridis')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_data(transaction_path: str, outer_path: str) -> None:
    # 處理交易數據，產生並儲存MTF影像。
    labels: List[str] = ['up_trend', 'no_up_trend']
    for label in labels:
        label_folder: str = os.path.join(outer_path, label)
        os.makedirs(label_folder, exist_ok=True)

    for ticker_folder in filter(lambda x: x[:3] in ['MXF', 'TXF'], os.listdir(transaction_path)):
        ticker_path: str = os.path.join(transaction_path, ticker_folder)
        print(f'Future ID: {ticker_folder}, MTF transforming on Transaction data Start')

        for file_name in os.listdir(ticker_path):
            file_path: str = os.path.join(ticker_path, file_name)
            data: pd.DataFrame = pd.read_csv(file_path)
            label: str = data['Trend'].iloc[0]

            if len(data) > 2:
                image_size: int = len(data)
                mtf_price: np.ndarray = generate_mtf_image(data['Price'], image_size)
                mtf_quantity: np.ndarray = generate_mtf_image(data['Quantity'], image_size)

                combined_mtf_image: np.ndarray = np.stack((mtf_price, mtf_quantity, np.zeros_like(mtf_price)), axis=-1)
                image_file_path: str = os.path.join(outer_path, label, f"{os.path.splitext(file_name)[0]}.png")
                save_image(combined_mtf_image, image_file_path)

        print(f'Future ID: {ticker_folder}, MTF transforming on Transaction data End')


def split_data_set(data_path: str, train_folder: str, val_folder: str, test_folder: str, train_ratio: float = 0.8,
                   val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    # 根據給定比例分割資料集為訓練集、驗證集和測試集。
    categories: List[str] = ['up_trend', 'down_trend', 'no_trend']
    for category in categories:
        category_path: str = os.path.join(data_path, category)
        images: List[str] = os.listdir(category_path)
        random.shuffle(images)

        num_train: int = int(len(images) * train_ratio)
        num_val: int = int(len(images) * val_ratio)

        for i, img in enumerate(images):
            src_path: str = os.path.join(category_path, img)
            if i < num_train:
                dest_path: str = os.path.join(train_folder, category, img)
            elif i < num_train + num_val:
                dest_path: str = os.path.join(val_folder, category, img)
            else:
                dest_path: str = os.path.join(test_folder, category, img)
            shutil.copy(src_path, dest_path)

        print(f'Category: {category}, Images allocated to training, validation, and test sets.')


if __name__ == "__main__":
    transaction_all_path: str = "..."
    img_all_path: str = "..."
    trainer_data_path: str = "..."
    process_data(transaction_all_path, img_all_path)
    split_data_set(img_all_path, os.path.join(trainer_data_path, 'train_img'),
                   os.path.join(trainer_data_path, 'val_img'), os.path.join(trainer_data_path, 'test_img'))
