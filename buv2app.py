import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# 1. 系統參數設定區
# ==========================================
st.set_page_config(page_title="ETH 模擬交易競賽戰情室", layout="wide", page_icon="🏆")

# Google Sheets CSV 網址
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXKr7zhxm9ghhZnBKiX0WaUlHtXDd4ros3uIFjKHrf88ojtxxmc2klc0s5x2JD_QhgNHbnu7PEwCO3/pub?gid=1677608980&single=true&output=csv"

# ==========================================
# 2. 數據獲取與清洗引擎
# ==========================================
@st.cache_data(ttl=300)
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

@st.cache_data(ttl=3600)
def fetch_crypto_klines(symbol='ETH-USD', period='30d', interval='1h'):
    """
    使用 yfinance 替換 ccxt 以解決 Streamlit Cloud 地區封鎖問題
    """
    try:
        # 獲取雅虎財經數據
        df_k = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df_k.empty:
            st.error("獲取 K 線資料為空，請檢查代碼或網路。")
            return pd.DataFrame()

        # 處理 yfinance 新版 MultiIndex 問題 (若有)
        if isinstance(df_k.columns, pd.MultiIndex):
            df_k.columns = df_k.columns.get_level_values(0)

        # 統一欄位名稱為小寫，相容原有邏輯
        df_k.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # 時區對齊：轉換為台灣時間，並移除時區標示 (使其與 Google Sheets 解析的時間格式一致)
        df_k.index = df_k.index.tz_convert('Asia/Taipei').tz_localize(None)
        df_k.index.name = 'timestamp'
        
        return df_k
    except Exception as e:
        st.error(f"獲取市場數據失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 撤單匹配邏輯
# ==========================================
def process_cancellations(df):
    if df.empty: return df
    df['系統狀態'] = '待處理'
    for i in range(len(df)):
        action = str(df.loc[i, '直接進場/預掛價格/撤單'])
        if '撤單' in action:
            df.loc[i, '系統狀態'] = '執行撤單'
            user = df.loc[i, '參賽者']
            cancel_price = df.loc[i, '進場價位']
            for j in range(i-1, -1, -1):
                if (df.loc[j, '參賽者'] == user) and (df.loc[j, '進場價位'] == cancel_price) and ('預掛' in str(df.loc[j, '直接進場/預掛價格/撤單'])) and (df.loc[j, '系統狀態'] == '待處理'):
                    df.loc[j, '系統狀態'] = '已主動撤銷'
                    break
    return df

# ==========================================
# 4. 核心回測引擎
# ==========================================
def run_backtest(df_trades, df_kline):
    # 建立新欄位存放結果
    df_trades['實際進場時間'] = pd.NaT
    df_trades['實際離場時間'] = pd.NaT
    df_trades['最終結果'] = df_trades['系統狀態'] # 預設繼承撤單狀態
    
    for i, row in df_trades.iterrows():
        if row['系統狀態'] != '待處理': continue # 已經被撤單或非待處理的直接跳過
            
        record_time = row['時間戳記']
        entry_price = float(row['進場價位'])
        sl = float(row['停損'])
        tp = float(row['停利'])
        direction = str(row['多/ 空']).strip()
        action_type = str(row['直接進場/預掛價格/撤單'])
        
        actual_entry = None
        actual_exit = None
        result = "未成交 (掛單中)"
        
        # 1. 找進場點
        if '直接進場' in action_type:
            # 簡化邏輯：直接進場就以當下 K 線時間為準
            closest_kline_time = df_kline[df_kline.index >= record_time].index.min()
            actual_entry = closest_kline_time if pd.notnull(closest_kline_time) else record_time
        else:
            # 預掛單：往後找觸碰點
            future_klines = df_kline[df_kline.index >= record_time]
            for t, k in future_klines.iterrows():
                if k['low'] <= entry_price <= k['high']:
                    actual_entry = t
                    break
        
        # 2. 找離場點 (如果有進場)
        if actual_entry is not None:
            result = "持倉中"
            # 從進場的下一根 K 線開始找停損停利
            exit_klines = df_kline[df_kline.index > actual_entry]
            for t, k in exit_klines.iterrows():
                hit_sl, hit_tp = False, False
                if '空' in direction:
                    if k['high'] >= sl: hit_sl = True
                    if k['low'] <= tp: hit_tp = True
                elif '多' in direction:
                    if k['low'] <= sl: hit_sl = True
                    if k['high'] >= tp: hit_tp = True
                
                if hit_sl or hit_tp:
                    actual_exit = t
                    if hit_sl and hit_tp: result = '負 (插針雙殺算停損)' # 保守估計
                    elif hit_sl: result = '負 (停損/SL)'
                    elif hit_tp: result = '勝 (停利/TP)'
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
    # 處理數據
    df_processed = process_cancellations(df_raw.copy())
    df_result = run_backtest(df_processed, df_kline)
    
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
        
        # 計算平均持倉時間
        finished_trades = user_df[user_df['實際離場時間'].notnull()]
        if not finished_trades.empty:
            avg_duration = (finished_trades['實際離場時間'] - finished_trades['實際進場時間']).mean()
            avg_duration_str = f"{avg_duration.components.days}天 {avg_duration.components.hours}小時"
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
