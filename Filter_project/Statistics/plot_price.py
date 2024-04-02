import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


file_path: str = '...'
result_df_corrected: pd.DataFrame = pd.read_csv(file_path, header=0)

# 設定繪圖風格
sns.set(style="whitegrid")

# 定義bins範圍為-100到100，藉此觀察較細緻的數據
bins_range: List[int] = list(range(-100, 101, 1))

# 繪製直方圖，使用定義的bins範圍
plt.figure(figsize=(15, 6))
sns.histplot(data=result_df_corrected, x='Price Difference TWD', bins=bins_range, kde=False)
plt.title('Distribution of Price Differences within ±100')
plt.xlabel('Price Difference TWD')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  # 如有需要，旋轉x軸標籤以改善可讀性
plt.tight_layout()  # 調整佈局以適應標籤
save_path: str = '...'
plt.savefig(save_path)
plt.show()
