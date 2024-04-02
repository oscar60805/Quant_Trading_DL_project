import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score


def compute_dataset_statistics(root_dir, class_names):
    scaler = StandardScaler()
    data_list = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith('.csv'):
                path = os.path.join(class_dir, fname)
                # 加载CSV文件
                data = pd.read_csv(path, header=0).values
                data_list.append(data)
    all_data = np.concatenate(data_list, axis=0)
    scaler.fit(all_data)
    mean = scaler.mean_
    scale = scaler.scale_
    return mean, scale


# 試著使用與訓練及相同的標準化方法，觀察最大泛化的效果

val_data_path = "/home/oscar/AllKindData/7500tick_candlestick_TransactionS/Merged_Training_Data/train"
class_names = ['no_up_trend', 'up_trend']  # 順序沒差，因為是計算統計量
val_mean, val_scale = compute_dataset_statistics(val_data_path, class_names)


class CNN1DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1DModel, self).__init__()
        # 卷积层输出通道数修改为8
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=1, padding=1)

        # 更新后，根据最大池化层的kernel size和stride为4，重新计算length_after_pool
        length_after_conv = (7500 + 2 * 1 - 2) // 1 + 1  # 卷积操作后的长度，这里的公式是基于padding和kernel size的标准计算公式
        length_after_pool = length_after_conv // 4  # 池化操作后的长度

        # 最后全连接层的参数从原来的512修改为256
        self.fc1 = nn.Linear(8 * length_after_pool, 256)  # 注意这里的8是卷积层的输出通道数
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=4, stride=4)  # 维持池化层的kernel size和stride为4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CSVDataset(Dataset):
    def __init__(self, root_dir, transform=None, mean=None, scale=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.mean = mean
        self.scale = scale

        # 定義類別
        classes = ['up_trend', 'no_up_trend']
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

        # 遍歷每個類別的資料夾，收集檔案路徑和標籤
        for class_name in classes:
            class_idx = class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith('.csv'):  # 確保是CSV文件
                    path = os.path.join(class_dir, fname)
                    item = (path, class_idx)
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # 加载CSV文件
        data = pd.read_csv(path, header=0)
        # 将数据转换为numpy数组
        data = data.values
        # 标准化处理
        if self.mean is not None and self.scale is not None:
            data = (data - self.mean) / self.scale
        # 转换为torch张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        # 调整形状以匹配1D CNN的输入要求 ([Length, Feature])
        data_tensor = data_tensor.transpose(0, 1)
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            data_tensor = self.transform(data_tensor)
        return data_tensor, label


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = CNN1DModel().to(self.device)
        self.load_model()
        self.test_iter = DataLoader(
            CSVDataset(root_dir=self.config['test_data_path'], mean=val_mean, scale=val_scale),
            batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        self.class_names = ['up_trend', 'no_up_trend']

    def load_model(self):
        self.net.load_state_dict(torch.load(self.config['model_path']))
        self.net.to(self.device)

    def analyze(self):
        self.net.eval()
        true_labels = []
        predictions = []
        predictionss = []

        filenames = [path.split('/')[-1] for path, _ in self.test_iter.dataset.samples]

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_iter, desc="Processing", leave=True):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)

                # 收集真實標籤和預測標籤
                true_labels.extend(labels.cpu().numpy())
                predictionss.extend(predicted.cpu().numpy())

                predictions.extend([self.class_names[p] for p in predicted.cpu().numpy()])

        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictionss)
        print("Confusion Matrix:")
        print(cm)

        # 計算精確率
        precision = precision_score(true_labels, predictionss, average='binary')  # 'binary'適用於二分類問題
        print(f"Precision: {precision:.2f}")

        # 計算每個類別的精確率
        precision_each_class = precision_score(true_labels, predictionss, average=None)  # 傳回每個類別的精確率
        print(f"Precision for each class: {precision_each_class}")

        class_mapping = {'up_trend': 'Do action', 'no_up_trend': 'No Action'}
        predictions = [class_mapping[pred] if pred in class_mapping else pred for pred in predictions]

        # 建立DataFrame並儲存
        prediction_df = pd.DataFrame({"Futures's ID _ Timestamp": filenames, "Prediction": predictions})
        prediction_df.sort_values(by="Futures's ID _ Timestamp", inplace=True)  # 根據指定列排序
        os.makedirs(self.config['Prediction_csv_save_path'], exist_ok=True)
        prediction_df.to_csv(
            os.path.join(self.config['Prediction_csv_save_path'], 'XXX_prediction.csv'),
            index=False)


def main():
    config = {
        'batch_size': 32,
        'test_data_path': "...",
        'model_path': '...',
        'Prediction_csv_save_path': '...'
    }

    # 加載模型並進行錯誤分析
    start_time = time.time()
    analyzer = Tester(config)
    analyzer.analyze()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
