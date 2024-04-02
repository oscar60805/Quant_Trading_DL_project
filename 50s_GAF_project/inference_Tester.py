import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score


class Model(nn.Module):
    def __init__(self, num_classes=2):
        ...

    def forward(self, x):
        ...


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = Model().to(self.device)
        self.load_model()
        self.criterion = nn.CrossEntropyLoss()
        self.test_iter = self.load_dataset()

    def load_model(self):
        self.net.load_state_dict(torch.load(self.config['model_path']))
        self.net.to(self.device)

    def load_dataset(self):
        test_transform = transforms.Compose([
            transforms.Resize((228, 228)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_set = torchvision.datasets.ImageFolder(self.config['test_data_path'], transform=test_transform)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size=self.config['batch_size'], shuffle=False,
                                                num_workers=4)
        self.class_names = test_set.classes
        print(self.class_names, len(self.class_names))
        return test_iter

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
        'model_path': '/...',
        'Prediction_csv_save_path': '...'
    }

    # 加載模型並進行錯誤分析
    start_time = time.time()
    analyzer = Tester(config)
    analyzer.analyze()
    end_time = time.time()
    total_time = end_time - start_time

    # 計算小時、分鐘和秒
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 打印結果
    print(f"Total testing time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")


if __name__ == '__main__':
    main()
