import pandas as pd

# 載入提供的CSV文件
file_path: str = '...'
df: pd.DataFrame = pd.read_csv(file_path, header=0)

# 確認轉換前的Date/Time格式
sample_date_time: str = df['Date/Time'][0]
print(f"Sample Date/Time: {sample_date_time}")

# 調整轉換方法並套用
df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
results: list = []

for trade in df['Trade #'].unique():
    entry: pd.DataFrame = df[(df['Trade #'] == trade) & (df['Type'].str.contains('Entry'))]
    exit: pd.DataFrame = df[(df['Trade #'] == trade) & (df['Type'].str.contains('Exit'))]

    if not entry.empty and not exit.empty:
        price_diff: float = exit['Price TWD'].values[0] - entry['Price TWD'].values[0]
        duration: float = (exit['Date/Time'].iloc[0] - entry['Date/Time'].iloc[0]) / pd.Timedelta(minutes=1)
    else:
        price_diff = None
        duration = None

    results.append({'Trade': trade, 'Price Difference TWD': price_diff, 'Duration (Minutes)': duration})

result_df_corrected: pd.DataFrame = pd.DataFrame(results)
result_df_corrected.to_csv('...', index=False)
