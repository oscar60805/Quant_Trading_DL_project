import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

file_path: str = '...'
result_df_corrected: pd.DataFrame = pd.read_csv(file_path, header=0)

# 定義bins範圍為0到400，
bins_range: List[int] = list(range(0, 401, 1))

plt.figure(figsize=(15, 6))
sns.histplot(data=result_df_corrected, x='Duration (Minutes)', bins=bins_range, kde=False)
plt.title('交易持續時間分佈 (0至400分鐘)')
plt.xlabel('持續時間 (分鐘)')
plt.ylabel('頻率')
plt.xticks(rotation=90)
plt.tight_layout()

save_path: str = '...'
plt.savefig(save_path)
plt.show()
