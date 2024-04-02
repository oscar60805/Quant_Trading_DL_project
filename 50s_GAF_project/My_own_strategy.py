import pandas as pd
import os


def profit_calculator(spath, tpath):
    df = pd.read_csv(spath)
    strategy_list = df[df['Prediction'] == 'Do action']["Futures's ID _ Timestamp"]
    txf_count = 0
    total_profit = 0
    total_win_numbers = 0
    trade_list = []
    for strategy in strategy_list:
        id = strategy[:-20]
        if id[0:3] != 'TXF':
            continue
        txf_count += 1
        true_data_path = os.path.join(tpath, id, strategy)
        true_data = pd.read_csv(true_data_path)
        trade = true_data.iloc[0]['if_trade']
        trade_list.append(trade)
        total_profit += trade
        if trade > 0:
            total_win_numbers += 1

    # 計算累積收益
    cumulative_profits = [sum(trade_list[:i + 1]) for i in range(len(trade_list))]

    # 計算最大回撤
    peak = cumulative_profits[0]
    max_drawdown = 0
    for profit in cumulative_profits:
        if profit > peak:  # 更新峰值
            peak = profit
        drawdown = peak - profit
        if drawdown > max_drawdown:  # 更新最大回撤
            max_drawdown = drawdown
    print(f'Every Trade Profit {trade_list}')
    print(f'Total Number of Transaction {txf_count}')
    print(f'Total Profit : {total_profit * 200}')
    print(f'Win Rate : {total_win_numbers / txf_count}')
    print(f'Max Drawdown : {max_drawdown * 200}')


if __name__ == '__main__':
    my_strategy_path = '...'
    data_path = '...'
    profit_calculator(my_strategy_path, data_path)
