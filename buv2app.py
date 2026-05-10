import re
from datetime import timedelta
import pandas as pd
import numpy as np

def run_comprehensive_backtest(df_trades, df_kline):
    """
    整合撤單、過期判定與倉位上限的一體化回測引擎
    """
    # 1. 初始化與排序
    df_trades = df_trades.sort_values('時間戳記').copy().reset_index(drop=True)
    df_trades['實際進場時間'] = pd.NaT
    df_trades['實際離場時間'] = pd.NaT
    df_trades['過期時間'] = pd.NaT
    df_trades['最終結果'] = '待處理'

    users = df_trades['參賽者'].dropna().unique()

    # 內部函數：推進時間線並結算有效訂單
    def update_active_trade(idx, up_to_time):
        row = df_trades.loc[idx]
        if row['最終結果'] not in ['未成交 (掛單中)', '待處理', '持倉中']:
            return

        record_time = row['時間戳記']
        expire_time = row['過期時間']
        entry_price = float(row['進場價位'])
        sl = float(row['停損'])
        tp = float(row['停利'])
        direction = str(row['多/ 空']).strip()
        is_direct = '直接進場' in str(row['直接進場/預掛價格/撤單'])

        actual_entry = df_trades.at[idx, '實際進場時間']

        # 定義 K 線搜尋區間
        start_time = actual_entry if pd.notnull(actual_entry) else record_time
        # 若尚未進場，搜尋至 up_to_time 與 過期時間 的較早者；若已進場，則搜尋至 up_to_time
        end_time = min(up_to_time, expire_time) if pd.isnull(actual_entry) else up_to_time
        
        klines = df_kline[(df_kline.index >= start_time) & (df_kline.index <= end_time)]

        # 尋找進場
        if pd.isnull(actual_entry):
            if is_direct:
                actual_entry = klines.index[0] if not klines.empty else record_time
            else:
                for t, k in klines.iterrows():
                    if k['low'] <= entry_price <= k['high']:
                        actual_entry = t
                        break

            if actual_entry is not None:
                df_trades.at[idx, '實際進場時間'] = actual_entry
                df_trades.at[idx, '最終結果'] = '持倉中'
            elif end_time >= expire_time:
                df_trades.at[idx, '最終結果'] = '已過期 (Expiry)'
                df_trades.at[idx, '實際離場時間'] = expire_time
                return # 過期則終止後續判定

        # 尋找離場 (若有進場)
        if pd.notnull(df_trades.at[idx, '實際進場時間']):
            actual_entry = df_trades.at[idx, '實際進場時間']
            exit_klines = df_kline[(df_kline.index > actual_entry) & (df_kline.index <= up_to_time)]
            
            for t, k in exit_klines.iterrows():
                hit_sl, hit_tp = False, False
                if '空' in direction:
                    if k['high'] >= sl: hit_sl = True
                    if k['low'] <= tp: hit_tp = True
                elif '多' in direction:
                    if k['low'] <= sl: hit_sl = True
                    if k['high'] >= tp: hit_tp = True
                
                if hit_sl or hit_tp:
                    df_trades.at[idx, '實際離場時間'] = t
                    if hit_sl and hit_tp: df_trades.at[idx, '最終結果'] = '負 (插針雙殺算停損)'
                    elif hit_sl: df_trades.at[idx, '最終結果'] = '負 (停損/SL)'
                    elif hit_tp: df_trades.at[idx, '最終結果'] = '勝 (停利/TP)'
                    break

    # 2. 依序審核每位參賽者的行為軌跡
    for user in users:
        user_idx = df_trades[df_trades['參賽者'] == user].index
        active_trade_idx = None # 記錄該參賽者當前「唯一有效」的單號

        for i in user_idx:
            row = df_trades.loc[i]
            current_time = row['時間戳記']
            action = str(row['直接進場/預掛價格/撤單'])

            # (A) 推進歷史訂單狀態至現在時間
            if active_trade_idx is not None:
                update_active_trade(active_trade_idx, current_time)
                # 若更新後該單已結束 (平倉或過期)，則釋放佔用
                status = df_trades.at[active_trade_idx, '最終結果']
                if status not in ['未成交 (掛單中)', '待處理', '持倉中']:
                    active_trade_idx = None

            # (B) 處理撤單請求
            if '撤單' in action:
                df_trades.at[i, '最終結果'] = '撤單操作紀錄'
                if active_trade_idx is not None:
                    active_row = df_trades.loc[active_trade_idx]
                    # 精準匹配：必須是預掛單、尚未成交，且進場價位相同
                    if ('預掛' in str(active_row['直接進場/預掛價格/撤單']) and 
                        float(active_row['進場價位']) == float(row['進場價位']) and
                        df_trades.at[active_trade_idx, '最終結果'] in ['待處理', '未成交 (掛單中)']):
                        
                        df_trades.at[active_trade_idx, '最終結果'] = '已主動撤銷'
                        df_trades.at[active_trade_idx, '實際離場時間'] = current_time
                        active_trade_idx = None # 撤單成功，釋放佔用
                continue

            # (C) 處理下單請求 (盲點防護：防止重疊下單)
            if active_trade_idx is not None:
                # 攔截！代表上一單還在掛單中或持倉中
                df_trades.at[i, '最終結果'] = '無效單 (已有持倉或掛單中)'
                continue

            # (D) 建立新單並解析效期
            df_trades.at[i, '最終結果'] = '未成交 (掛單中)'
            if '直接進場' in action:
                # 給予 1 小時的判定寬容度
                df_trades.at[i, '過期時間'] = current_time + timedelta(hours=1) 
            else:
                # 動態提取天數 (Regex)
                match = re.search(r'(\d+)天', action)
                days = int(match.group(1)) if match else 1 # 防呆預設 1 天
                df_trades.at[i, '過期時間'] = current_time + timedelta(days=days)

            active_trade_idx = i # 佔用系統狀態

        # 迴圈結束後，把該參賽者最後一筆有效單推演至市場最新時間
        if active_trade_idx is not None:
            update_active_trade(active_trade_idx, df_kline.index.max())

    return df_trades
