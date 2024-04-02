import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix
import numpy as np


class ResNet50Model(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=3):
        super(ResNet50Model, self).__init__()
        # 這裡使用Pretrain好的ResNet50模型
        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 將最後的fully connect Layer 類別數改為2
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = ResNet50Model().to(self.device)
        self.train_iter, self.val_iter = self.load_dataset()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.75)
        self.record_train = []
        self.record_test = []

    def load_dataset(self):
        # 定義訓練集的數據增強以及轉換格式
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # 將 PIL 影像轉換為Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根據ImageNet的標準化參數進行標準化
                                 std=[0.229, 0.224, 0.225])
        ])

        # 驗證集只需數據轉換不需做增強
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_set = torchvision.datasets.ImageFolder(
            self.config['train_data_path'],
            transform=train_transform
        )
        val_set = torchvision.datasets.ImageFolder(
            self.config['val_data_path'],
            transform=val_transform
        )
        train_iter = torch.utils.data.DataLoader(
            train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=4
        )
        val_iter = torch.utils.data.DataLoader(
            val_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=4
        )
        return train_iter, val_iter

    def configure_optimizer(self):
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.config['learning_rate'],  # 學習率
            weight_decay=self.config['weight_decay']  # 權重衰減
        )
        return optimizer

    def train(self):
        self.net.train()
        num_print = len(self.train_iter) // 4

        for epoch in range(self.config['num_epochs']):
            print(f"========== number {epoch + 1} epoch training ==========")
            total, correct, train_loss = 0, 0, 0
            start = time.time()

            for i, (X, y) in tqdm(enumerate(self.train_iter)):
                X, y = X.to(self.device), y.to(self.device)
                output = self.net(X)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total += y.size(0)
                correct += (output.argmax(dim=1) == y).sum().item()
                train_acc = 100.0 * correct / total

                if (i + 1) % num_print == 0:
                    print(
                        f"進度: [{i + 1}/{len(self.train_iter)}], 訓練損失: {train_loss / (i + 1):.3f} | 訓練準確度: {train_acc:6.3f}% | 學習率: {self.get_cur_lr()}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"花費時間: {time.time() - start:.4f}秒")

            if self.val_iter is not None:
                self.record_test.append(self.validate())
            self.record_train.append(train_acc)
            torch.save(self.net.state_dict(
            ), os.path.join(self.config['weight_path'],
                            f"Resnet50_modelTrainer_{epoch + 1}_acc={self.record_test[epoch]:.3f}.pt"))
            print("儲存權重完成")
        torch.save(self.net.state_dict(),
                   os.path.join(self.config['weight_path'], f"Resnet50_modelTrainer_full.pt"))

    def validate(self):
        total, correct = 0, 0
        total_loss = 0
        self.net.eval()
        all_labels = []
        all_probs = []
        all_preds = []
        adjusted_preds = []  # 儲存調整閾值後的預測

        with torch.no_grad():
            print("*************** validation ***************")
            for X, y in tqdm(self.val_iter):
                X, y = X.to(self.device), y.to(self.device)
                output = self.net(X)
                _, preds = torch.max(output, 1)
                loss = self.val_criterion(output, y)

                total_loss += loss.item()  # 累加每個批次的損失

                probs = F.softmax(output, dim=1)[:, 1]  # 取得屬於類別1的機率
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())  # 累積預測結果

                adjusted_pred = (probs > self.config['threshold']).long().cpu().numpy()
                adjusted_preds.extend(adjusted_pred)

                total += y.size(0)
                correct += (preds == y).sum().item()

            avg_val_loss = total_loss / len(self.val_iter)  # 計算整個驗證集上的平均損失
            val_acc = 100.0 * correct / total
            print(f"validation loss: {avg_val_loss:.3f} | validation accuracy: {val_acc:6.3f}%")
            print('=========================\n')

            # 計算並列印調整閾值後的準確率
            adjusted_acc = 100.0 * np.sum(np.array(all_labels) == np.array(adjusted_preds)) / len(all_labels)
            print(f"Adjusted validation accuracy (threshold={self.config['threshold']}): {adjusted_acc:6.3f}%")
            print('=========================\n')

            # 計算混淆矩陣
            cm = confusion_matrix(all_labels, all_preds)
            print("Confusion Matrix:")
            print(cm)
            print()

            # 印出混淆矩陣的各項指標（真正率、假正率、真負率、假負率）
            TN, FP, FN, TP = cm.ravel()
            print(f"True Negatives: {TN}")
            print(f"False Positives: {FP}")
            print(f"False Negatives: {FN}")
            print(f"True Positives: {TP}")
            print('=========================\n')

            # 二分類的AUC計算
            auc_score = roc_auc_score(all_labels, all_probs)
            print(f"Validation AUC: {auc_score:.3f}")
            print('=========================\n')

            # 計算每個類別的精確度
            labels = [0, 1]
            precision = precision_score(all_labels, all_preds, average=None, labels=labels)
            class_names = ['up_trend', 'no_up_trend']
            for i, class_name in enumerate(class_names):
                print(f"{class_name} Precision: {precision[i]:.3f}")

            print("************************************\n")
        self.net.train()
        return val_acc

    def get_cur_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def learning_curve(self):
        plt.style.use("ggplot")
        plt.plot(range(1, len(self.record_train) + 1),
                 self.record_train, label="Training ACC")
        if self.record_test:
            plt.plot(range(1, len(self.record_test) + 1),
                     self.record_test, label="Validation ACC")
        plt.legend(loc=4)
        plt.title("Learning Curve")
        plt.xticks(range(0, len(self.record_train) + 1, 5))
        plt.yticks(range(0, 101, 5))
        plt.xlabel("Nums of Epoch")
        plt.ylabel("ACC")

        # 保存學習曲線圖像
        learning_curve_plt_path = os.path.join(self.config['weight_path'], 'learning_curve.png')
        plt.savefig(learning_curve_plt_path)
        plt.close()


def main():
    config = {
        'train_data_path': "...",
        'val_data_path': "...",
        'weight_path': "...",
        'batch_size': 32,
        'num_epochs': 300,
        'learning_rate': 0.000125,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_classes': 2
    }
    os.makedirs(config['weight_path'], exist_ok=True)
    trainer = Trainer(config)
    start_time = time.time()
    trainer.train()
    trainer.learning_curve()
    end_time = time.time()
    total_time = end_time - start_time

    # 計算小時、分鐘和秒
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 打印結果
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")


if __name__ == '__main__':
    main()
