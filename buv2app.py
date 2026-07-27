import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re


# ==========================================
# 1. 系統參數設定區
# ==========================================
st.set_page_config(page_title="ETH 模擬交易競賽戰情室", layout="wide", page_icon="🏆")

# Google Sheets CSV 網址
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXKr7zhxm9ghhZnBKiX0WaUlHtXDd4ros3uIFjKHrf88ojtxxmc2klc0s5x2JD_QhgNHbnu7PEwCO3/pub?gid=1677608980&single=true&output=csv"

# ==========================================
# 2. 數據獲取與清洗引擎
# ==========================================
# @st.cache_data(ttl=300)
def load_and_clean_data(url):
    try:
        df = pd.read_csv(url)
        # --- 解決時間解析錯誤 (終極版) ---
        def fix_tw_time(t_str):
            t_str = str(t_str)
            if '上午' in t_str: return t_str.replace('上午 ', '') + ' AM'
            elif '下午' in t_str: return t_str.replace('下午 ', '') + ' PM'
            return t_str
            
        df['時間戳記'] = df['時間戳記'].apply(fix_tw_time)
        df['時間戳記'] = pd.to_datetime(df['時間戳記'], format='mixed')
        
        df = df.sort_values('時間戳記').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"讀取 Google Sheets 失敗: {e}")
        return pd.DataFrame()

# @st.cache_data(ttl=600) # 既然起點固定，ttl 可以稍微拉長，節省資源
def fetch_crypto_klines(symbol='ETH-USD'):
    try:
        # 💡 使用固定起點，模擬測試腳本成功的路徑
        # 注意：我們不設 end，代表抓到「現在」為止
        START_DATE = '2026-04-20' 
        
        df_k = yf.download(symbol, start=START_DATE, interval='1h', progress=False)
        
        if df_k.empty:
            return pd.DataFrame()

        # 1. 統一欄位名稱
        if isinstance(df_k.columns, pd.MultiIndex):
            df_k.columns = df_k.columns.get_level_values(0)
        df_k.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        # 2. 時區處理 (UTC -> 台北)
        df_k.index = df_k.index.tz_convert('Asia/Taipei').tz_localize(None)
        
        # 3. 核心加固：強迫對齊時間格 (即使 API 漏給資料，格子也要在)
        full_range = pd.date_range(start=df_k.index.min(), end=df_k.index.max(), freq='h')
        df_k = df_k.reindex(full_range)
        
        # 💡 數據分析師建議：不使用 ffill，保持 NaN，讓「證據分析」能抓出 API 缺漏
        return df_k
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return pd.DataFrame()




# ==========================================
# 3. 撤單匹配邏輯
# ==========================================
def run_comprehensive_backtest(df_trades, df_kline):
    """
    整合撤單、過期判定與倉位上限的一體化回測引擎

    直接進場規則：
    - 使用者回報「直接進場」時，視為宣稱已在回報時間成交。
    - 系統用「回報時間前 1 小時 ～ 回報時間後 1 小時」的市場價格，
      驗證申報進場價是否合理曾被觸及。
    - 若有觸價，實際進場時間仍記錄「使用者回報時間」。
    - 若回報後 1 小時結束仍完全未觸價，才判定為
      「未成交 (直接進場前後1小時未觸價)」。
    """
    df_trades = df_trades.sort_values('時間戳記').copy().reset_index(drop=True)
    df_trades['實際進場時間'] = pd.NaT
    df_trades['實際離場時間'] = pd.NaT
    df_trades['過期時間'] = pd.NaT
    df_trades['最終結果'] = '待處理'

    users = df_trades['參賽者'].dropna().unique()

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
        action = str(row['直接進場/預掛價格/撤單'])
        is_direct = '直接進場' in action

        actual_entry = df_trades.at[idx, '實際進場時間']

        # ==========================================================
        # A. 尋找 / 驗證進場
        # ==========================================================
        if pd.isnull(actual_entry):

            if is_direct:
                # 直接進場：回報時間前後各 1 小時做價格合理性驗證
                validation_start = record_time - timedelta(hours=1)
                validation_end = min(
                    up_to_time,
                    record_time + timedelta(hours=1)
                )
            else:
                # 預掛單：只能從下單時間往後找，最多找到掛單效期
                validation_start = record_time
                validation_end = min(up_to_time, expire_time)

            klines = df_kline[
                (df_kline.index >= validation_start) &
                (df_kline.index <= validation_end)
            ]

            valid_klines = klines.dropna(subset=['low', 'high'])

            price_touched = False
            touch_time = None

            for t, k in valid_klines.iterrows():
                if k['low'] <= entry_price <= k['high']:
                    price_touched = True
                    touch_time = t
                    break

            if price_touched:
                if is_direct:
                    # K 線只是驗證申報價格合理性；
                    # 真正進場時間仍記錄使用者的回報時間。
                    actual_entry = record_time
                else:
                    # 預掛單以第一次觸價時間作為進場時間。
                    actual_entry = touch_time

                df_trades.at[idx, '實際進場時間'] = actual_entry
                df_trades.at[idx, '最終結果'] = '持倉中'

            else:
                if is_direct:
                    direct_validation_end = record_time + timedelta(hours=1)

                    if up_to_time >= direct_validation_end:
                        df_trades.at[idx, '最終結果'] = '未成交 (直接進場前後1小時未觸價)'
                        df_trades.at[idx, '實際離場時間'] = direct_validation_end
                        return

                    # 尚未走完驗證視窗，維持掛單中，之後繼續驗證
                    df_trades.at[idx, '最終結果'] = '未成交 (掛單中)'
                    return

                else:
                    if validation_end >= expire_time:
                        df_trades.at[idx, '最終結果'] = '已過期 (Expiry)'
                        df_trades.at[idx, '實際離場時間'] = expire_time
                        return

                    df_trades.at[idx, '最終結果'] = '未成交 (掛單中)'
                    return

        # ==========================================================
        # B. 尋找離場
        # ==========================================================
        if pd.notnull(df_trades.at[idx, '實際進場時間']):
            actual_entry = df_trades.at[idx, '實際進場時間']

            exit_klines = df_kline[
                (df_kline.index > actual_entry) &
                (df_kline.index <= up_to_time)
            ].dropna(subset=['low', 'high'])

            for t, k in exit_klines.iterrows():
                hit_sl, hit_tp = False, False

                if '空' in direction:
                    if k['high'] >= sl:
                        hit_sl = True
                    if k['low'] <= tp:
                        hit_tp = True

                elif '多' in direction:
                    if k['low'] <= sl:
                        hit_sl = True
                    if k['high'] >= tp:
                        hit_tp = True

                if hit_sl or hit_tp:
                    df_trades.at[idx, '實際離場時間'] = t

                    if hit_sl and hit_tp:
                        df_trades.at[idx, '最終結果'] = '負 (插針雙殺算停損)'
                    elif hit_sl:
                        df_trades.at[idx, '最終結果'] = '負 (停損/SL)'
                    elif hit_tp:
                        df_trades.at[idx, '最終結果'] = '勝 (停利/TP)'

                    break

    # 依序審核每位參賽者的行為軌跡
    for user in users:
        user_idx = df_trades[df_trades['參賽者'] == user].index
        active_trade_idx = None

        for i in user_idx:
            row = df_trades.loc[i]
            current_time = row['時間戳記']
            action = str(row['直接進場/預掛價格/撤單'])

            # 推進上一筆有效訂單
            if active_trade_idx is not None:
                update_active_trade(active_trade_idx, current_time)

                status = df_trades.at[active_trade_idx, '最終結果']
                if status not in ['未成交 (掛單中)', '待處理', '持倉中']:
                    active_trade_idx = None

            # 處理撤單
            if '撤單' in action:
                df_trades.at[i, '最終結果'] = '撤單操作紀錄'

                if active_trade_idx is not None:
                    status = str(df_trades.at[active_trade_idx, '最終結果'])

                    if '未成交' in status or '待處理' in status:
                        df_trades.at[active_trade_idx, '最終結果'] = '已主動撤銷'
                        df_trades.at[active_trade_idx, '實際離場時間'] = current_time
                        active_trade_idx = None

                continue

            # 防止重疊下單
            if active_trade_idx is not None:
                df_trades.at[i, '最終結果'] = '無效單 (已有持倉或掛單中)'
                continue

            # 建立新單
            df_trades.at[i, '最終結果'] = '未成交 (掛單中)'

            if '直接進場' in action:
                # 回報後 1 小時才完成「前後各 1 小時」的驗證視窗
                df_trades.at[i, '過期時間'] = current_time + timedelta(hours=1)
            else:
                match = re.search(r'(\d+)天', action)
                days = int(match.group(1)) if match else 1
                df_trades.at[i, '過期時間'] = current_time + timedelta(days=days)

            active_trade_idx = i

        # 最後一筆有效單推演到市場最新時間
        if active_trade_idx is not None:
            update_active_trade(active_trade_idx, df_kline.index.max())

    return df_trades

# ==========================================
# 4. 核心回測引擎
# ==========================================
def run_backtest(df_trades, df_kline):
    """
    舊版獨立回測函數。
    UI 目前使用 run_comprehensive_backtest()，
    但這裡同步套用相同的直接進場驗證規則。
    """
    df_trades['實際進場時間'] = pd.NaT
    df_trades['實際離場時間'] = pd.NaT
    df_trades['最終結果'] = df_trades['系統狀態']

    for i, row in df_trades.iterrows():
        if row['系統狀態'] != '待處理':
            continue

        record_time = row['時間戳記']
        entry_price = float(row['進場價位'])
        sl = float(row['停損'])
        tp = float(row['停利'])
        direction = str(row['多/ 空']).strip()
        action_type = str(row['直接進場/預掛價格/撤單'])

        actual_entry = None
        actual_exit = None
        result = "未成交 (掛單中)"

        # 1. 找 / 驗證進場點
        if '直接進場' in action_type:
            validation_start = record_time - timedelta(hours=1)
            validation_end = record_time + timedelta(hours=1)

            validation_klines = df_kline[
                (df_kline.index >= validation_start) &
                (df_kline.index <= validation_end)
            ].dropna(subset=['low', 'high'])

            price_touched = any(
                k['low'] <= entry_price <= k['high']
                for _, k in validation_klines.iterrows()
            )

            if price_touched:
                actual_entry = record_time
            else:
                result = '未成交 (直接進場前後1小時未觸價)'

        else:
            future_klines = df_kline[
                df_kline.index >= record_time
            ].dropna(subset=['low', 'high'])

            for t, k in future_klines.iterrows():
                if k['low'] <= entry_price <= k['high']:
                    actual_entry = t
                    break

        # 2. 找離場點
        if actual_entry is not None:
            result = "持倉中"

            exit_klines = df_kline[
                df_kline.index > actual_entry
            ].dropna(subset=['low', 'high'])

            for t, k in exit_klines.iterrows():
                hit_sl, hit_tp = False, False

                if '空' in direction:
                    if k['high'] >= sl:
                        hit_sl = True
                    if k['low'] <= tp:
                        hit_tp = True

                elif '多' in direction:
                    if k['low'] <= sl:
                        hit_sl = True
                    if k['high'] >= tp:
                        hit_tp = True

                if hit_sl or hit_tp:
                    actual_exit = t

                    if hit_sl and hit_tp:
                        result = '負 (插針雙殺算停損)'
                    elif hit_sl:
                        result = '負 (停損/SL)'
                    elif hit_tp:
                        result = '勝 (停利/TP)'

                    break

        df_trades.at[i, '實際進場時間'] = actual_entry
        df_trades.at[i, '實際離場時間'] = actual_exit
        df_trades.at[i, '最終結果'] = result

    return df_trades

# ==========================================
# 5. UI 與 視覺化渲染
# ==========================================
st.title("🏆 ETH 模擬交易競賽戰情室")

# 獲取與運算數據
df_raw = load_and_clean_data(GOOGLE_SHEET_CSV_URL)
df_kline = fetch_crypto_klines() # 使用新的數據獲取函數

if not df_raw.empty and not df_kline.empty:
    # 處理數據：傳入兩個參數，並直接將結果賦值給 df_result
    df_result = run_comprehensive_backtest(df_raw.copy(), df_kline)
    
    # --- 模組 A：戰力排行榜 ---
    st.subheader("🔥 即時戰力排行榜")
    leaderboard_data = []
    users = df_result['參賽者'].dropna().unique()
    
    for u in users:
        user_df = df_result[df_result['參賽者'] == u]
        wins = len(user_df[user_df['最終結果'].str.contains('勝', na=False)])
        losses = len(user_df[user_df['最終結果'].str.contains('負', na=False)])
        cancel_count = len(user_df[user_df['最終結果'].str.contains('撤銷', na=False)])
        holding_count = len(user_df[user_df['最終結果'].str.contains('持倉中', na=False)])
        
        # 計算平均持倉時間 (加入雙重過濾與空值驗證防呆機制)
        finished_trades = user_df[user_df['實際離場時間'].notnull() & user_df['實際進場時間'].notnull()]
        
        if not finished_trades.empty:
            avg_duration = (finished_trades['實際離場時間'] - finished_trades['實際進場時間']).mean()
            
            # 確保平均值不是 NaT 才提取天數與小時
            if pd.notnull(avg_duration):
                avg_duration_str = f"{avg_duration.components.days}天 {avg_duration.components.hours}小時"
            else:
                avg_duration_str = "無"
        else:
            avg_duration_str = "無"

        leaderboard_data.append({
            "參賽者": u,
            "淨勝分 (勝-負)": wins - losses,
            "勝 / 負": f"{wins} / {losses}",
            "持倉中": holding_count,
            "主動撤單次數": cancel_count,
            "平均持倉時間": avg_duration_str
        })
    
    if leaderboard_data:
        df_leaderboard = pd.DataFrame(leaderboard_data).sort_values("淨勝分 (勝-負)", ascending=False).reset_index(drop=True)
        st.dataframe(df_leaderboard, use_container_width=True)
    
    st.divider()

    # --- 模組 B：個人覆盤 Dashboard ---
    st.subheader("🔍 個人交易軌跡覆盤")
    if len(users) > 0:
        selected_user = st.selectbox("選擇要覆盤的參賽者：", users)
        user_trades = df_result[df_result['參賽者'] == selected_user]
        
        # 繪製 Plotly K 線圖
        fig = go.Figure(data=[go.Candlestick(
            x=df_kline.index, open=df_kline['open'], high=df_kline['high'],
            low=df_kline['low'], close=df_kline['close'], name="ETH/USDT 1H"
        )])
        
        # 畫出使用者的進出場紀錄與區塊
        for _, row in user_trades.iterrows():
            if pd.notnull(row['實際進場時間']):
                entry_t = row['實際進場時間']
                exit_t = row['實際離場時間'] if pd.notnull(row['實際離場時間']) else df_kline.index[-1]
                entry_p = row['進場價位']
                direction = row['多/ 空']
                result = row['最終結果']
                reason = row.get('進場理由', '無')
                
                # 判斷顏色
                if '勝' in result: box_color = "rgba(0, 255, 0, 0.2)"; border_color = "green"
                elif '負' in result: box_color = "rgba(255, 0, 0, 0.2)"; border_color = "red"
                else: box_color = "rgba(255, 255, 0, 0.2)"; border_color = "yellow"
                
                # 畫持倉區間框框
                y_max, y_min = entry_p * 1.05, entry_p * 0.95 
                mask = (df_kline.index >= entry_t) & (df_kline.index <= exit_t)
                if not df_kline[mask].empty:
                    y_max, y_min = df_kline[mask]['high'].max(), df_kline[mask]['low'].min()
                    
                fig.add_shape(type="rect", x0=entry_t, y0=y_min, x1=exit_t, y1=y_max,
                              line=dict(color=border_color, width=1.5), fillcolor=box_color)
                
                # 畫進場箭頭
                symbol = 'triangle-up' if '多' in str(direction) else 'triangle-down'
                arrow_color = 'green' if '多' in str(direction) else 'red'
                fig.add_trace(go.Scatter(
                    x=[entry_t], y=[entry_p], mode='markers',
                    marker=dict(symbol=symbol, size=15, color=arrow_color, line=dict(width=1, color='white')),
                    name=f"{direction} ({entry_p})",
                    hovertemplate=f"結果: {result}<br>價位: {entry_p}<br>理由: {reason}<extra></extra>"
                ))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # --- 模組 C：詳細交易紀錄表 ---
        st.subheader("📝 原始交易日誌")
        display_cols = ['時間戳記', '參賽者', '多/ 空', '進場價位', '最終結果', '實際進場時間', '實際離場時間', '進場理由']
        existing_cols = [c for c in display_cols if c in df_result.columns]
        st.dataframe(user_trades[existing_cols], use_container_width=True)

else:
    st.warning("請等待市場數據讀取或確認 Google Sheet 來源有效。")
