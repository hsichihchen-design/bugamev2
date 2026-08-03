import re
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ==========================================
# 1. 系統參數設定區
# ==========================================
st.set_page_config(
    page_title="ETH 模擬交易競賽戰情室",
    layout="wide",
    page_icon="🏆",
)

# Google Sheets CSV 網址
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vTXKr7zhxm9ghhZnBKiX0WaUlHtXDd4ros3uIFjKHrf88ojtxxmc2klc0s5x2JD_QhgNHbnu7PEwCO3/"
    "pub?gid=1677608980&single=true&output=csv"
)

# 幣安 USDⓈ-M 永續合約市場資料
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
BINANCE_SYMBOL = "ETHUSDT"
KLINE_INTERVAL = "1h"
KLINE_START_TAIPEI = pd.Timestamp("2026-04-20 00:00:00")

# 比賽截止時間：台北時間 2026/07/31 23:59:59
# 7/31 23:59 仍屬比賽時間；8/1 00:00 起視為比賽已結束。
COMPETITION_END = pd.Timestamp("2026-07-31 23:59:59")
COMPETITION_END_LABEL = "2026/07/31 23:59（台北時間）"
COMPETITION_ENDED_STATUS = "比賽已結束"

REQUEST_TIMEOUT_SECONDS = 20
BINANCE_KLINE_LIMIT = 1500
INTERVAL_MILLISECONDS = {
    "1h": 60 * 60 * 1000,
}


# ==========================================
# 2. 數據獲取與清洗引擎
# ==========================================
@st.cache_data(ttl=60)
def load_and_clean_data(url):
    try:
        df = pd.read_csv(url)

        def fix_tw_time(t_str):
            t_str = str(t_str)
            if "上午" in t_str:
                return t_str.replace("上午 ", "") + " AM"
            if "下午" in t_str:
                return t_str.replace("下午 ", "") + " PM"
            return t_str

        df["時間戳記"] = df["時間戳記"].apply(fix_tw_time)
        df["時間戳記"] = pd.to_datetime(df["時間戳記"], format="mixed")
        df = df.sort_values("時間戳記").reset_index(drop=True)
        return df
    except Exception as exc:
        st.error(f"讀取 Google Sheets 失敗: {exc}")
        return pd.DataFrame()


def taipei_time_to_utc_ms(value):
    """將台北時間轉成幣安 API 使用的 UTC 毫秒時間戳。"""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Taipei")
    else:
        timestamp = timestamp.tz_convert("Asia/Taipei")
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


@st.cache_data(ttl=60)
def fetch_binance_klines(
    symbol=BINANCE_SYMBOL,
    interval=KLINE_INTERVAL,
    start_time=KLINE_START_TAIPEI,
):
    """
    取得幣安 USDⓈ-M Futures 的一般成交價 K 線。

    使用 /fapi/v1/klines，而不是 markPriceKlines，
    因此 OHLC 是 ETHUSDT 永續合約的實際成交價格。
    """
    try:
        interval_ms = INTERVAL_MILLISECONDS[interval]
        next_start_ms = taipei_time_to_utc_ms(start_time)
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        all_rows = []

        while next_start_ms <= end_ms:
            response = requests.get(
                f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": next_start_ms,
                    "endTime": end_ms,
                    "limit": BINANCE_KLINE_LIMIT,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            rows = response.json()

            if not isinstance(rows, list):
                raise ValueError(f"幣安 K 線回傳格式異常: {rows}")
            if not rows:
                break

            all_rows.extend(rows)
            last_open_ms = int(rows[-1][0])
            new_start_ms = last_open_ms + interval_ms

            if new_start_ms <= next_start_ms:
                raise RuntimeError("幣安 K 線分頁時間沒有前進，已停止避免無限迴圈。")

            next_start_ms = new_start_ms

            if len(rows) < BINANCE_KLINE_LIMIT:
                break

        if not all_rows:
            return pd.DataFrame()

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df_k = pd.DataFrame(all_rows, columns=columns)

        for column in ["open", "high", "low", "close", "volume"]:
            df_k[column] = pd.to_numeric(df_k[column], errors="coerce")

        df_k["open_time"] = (
            pd.to_datetime(df_k["open_time"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Taipei")
            .dt.tz_localize(None)
        )
        df_k = (
            df_k.set_index("open_time")[["open", "high", "low", "close", "volume"]]
            .sort_index()
        )
        df_k = df_k[~df_k.index.duplicated(keep="last")]

        # 保留缺漏為 NaN，不用前值填補，避免製造不存在的價格證據。
        full_range = pd.date_range(
            start=df_k.index.min(),
            end=df_k.index.max(),
            freq="h",
        )
        return df_k.reindex(full_range)

    except Exception as exc:
        st.error(f"幣安 K 線讀取失敗: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def fetch_binance_latest_price(symbol=BINANCE_SYMBOL):
    """取得幣安 USDⓈ-M Futures 最新成交價（Last Price），不是 Mark Price。"""
    try:
        response = requests.get(
            f"{BINANCE_FUTURES_BASE_URL}/fapi/v2/ticker/price",
            params={"symbol": symbol},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        latest_price = float(payload["price"])
        transaction_time = (
            pd.to_datetime(payload["time"], unit="ms", utc=True)
            .tz_convert("Asia/Taipei")
            .tz_localize(None)
        )
        return latest_price, transaction_time
    except Exception as exc:
        st.warning(f"幣安最新價格讀取失敗: {exc}")
        return None, None


# ==========================================
# 3. 撤單匹配與比賽截止邏輯
# ==========================================
def run_comprehensive_backtest(df_trades, df_kline):
    """
    整合撤單、過期判定、倉位上限與比賽截止的一體化回測引擎。

    比賽截止規則：
    - 只使用 2026/07/31 23:59:59（台北時間）以前的行情判定勝負。
    - 截止時間之後新增的紀錄，標示「比賽已結束」。
    - 截止時仍在掛單或持倉中的訂單，標示「比賽已結束」。

    直接進場規則：
    - 使用者回報「直接進場」時，視為宣稱已在回報時間成交。
    - 系統用回報時間前 1 小時到後 1 小時的 K 線驗證價格是否曾觸及。
    - 若有觸價，實際進場時間仍記錄使用者回報時間。
    - 若完整驗證區間內未觸價，判定為「未成交」。
    """
    df_trades = df_trades.sort_values("時間戳記").copy().reset_index(drop=True)
    df_trades["實際進場時間"] = pd.NaT
    df_trades["實際離場時間"] = pd.NaT
    df_trades["過期時間"] = pd.NaT
    df_trades["最終結果"] = "待處理"

    users = df_trades["參賽者"].dropna().unique()

    def mark_competition_ended(idx):
        df_trades.at[idx, "最終結果"] = COMPETITION_ENDED_STATUS
        df_trades.at[idx, "實際離場時間"] = COMPETITION_END

    def update_active_trade(idx, requested_up_to_time):
        row = df_trades.loc[idx]

        if row["最終結果"] not in ["未成交 (掛單中)", "待處理", "持倉中"]:
            return

        record_time = row["時間戳記"]
        if record_time > COMPETITION_END:
            mark_competition_ended(idx)
            return

        # 關鍵防線：任何訂單最多只推演到比賽截止時間。
        up_to_time = min(pd.Timestamp(requested_up_to_time), COMPETITION_END)

        expire_time = row["過期時間"]
        entry_price = float(row["進場價位"])
        sl = float(row["停損"])
        tp = float(row["停利"])
        direction = str(row["多/ 空"]).strip()
        action = str(row["直接進場/預掛價格/撤單"])
        is_direct = "直接進場" in action

        actual_entry = df_trades.at[idx, "實際進場時間"]

        # ==========================================================
        # A. 尋找 / 驗證進場
        # ==========================================================
        if pd.isnull(actual_entry):
            if is_direct:
                validation_start = record_time - timedelta(hours=1)
                full_validation_end = record_time + timedelta(hours=1)
                validation_end = min(up_to_time, full_validation_end)
            else:
                validation_start = record_time
                validation_end = min(up_to_time, expire_time)

            valid_klines = df_kline[
                (df_kline.index >= validation_start)
                & (df_kline.index <= validation_end)
            ].dropna(subset=["low", "high"])

            price_touched = False
            touch_time = None

            for candle_time, candle in valid_klines.iterrows():
                if candle["low"] <= entry_price <= candle["high"]:
                    price_touched = True
                    touch_time = candle_time
                    break

            if price_touched:
                actual_entry = record_time if is_direct else touch_time
                df_trades.at[idx, "實際進場時間"] = actual_entry
                df_trades.at[idx, "最終結果"] = "持倉中"
            else:
                if is_direct:
                    if up_to_time >= full_validation_end:
                        df_trades.at[idx, "最終結果"] = (
                            "未成交 (直接進場前後2小時未觸價)"
                        )
                        df_trades.at[idx, "實際離場時間"] = full_validation_end
                        return

                    if up_to_time >= COMPETITION_END:
                        mark_competition_ended(idx)
                        return

                    df_trades.at[idx, "最終結果"] = "未成交 (掛單中)"
                    return

                if validation_end >= expire_time:
                    df_trades.at[idx, "最終結果"] = "已過期 (Expiry)"
                    df_trades.at[idx, "實際離場時間"] = expire_time
                    return

                if up_to_time >= COMPETITION_END:
                    mark_competition_ended(idx)
                    return

                df_trades.at[idx, "最終結果"] = "未成交 (掛單中)"
                return

        # ==========================================================
        # B. 尋找離場
        # ==========================================================
        if pd.notnull(df_trades.at[idx, "實際進場時間"]):
            actual_entry = df_trades.at[idx, "實際進場時間"]
            exit_klines = df_kline[
                (df_kline.index > actual_entry)
                & (df_kline.index <= up_to_time)
            ].dropna(subset=["low", "high"])

            for candle_time, candle in exit_klines.iterrows():
                hit_sl = False
                hit_tp = False

                if "空" in direction:
                    hit_sl = candle["high"] >= sl
                    hit_tp = candle["low"] <= tp
                elif "多" in direction:
                    hit_sl = candle["low"] <= sl
                    hit_tp = candle["high"] >= tp

                if hit_sl or hit_tp:
                    df_trades.at[idx, "實際離場時間"] = candle_time

                    if hit_sl and hit_tp:
                        df_trades.at[idx, "最終結果"] = "負 (插針雙殺算停損)"
                    elif hit_sl:
                        df_trades.at[idx, "最終結果"] = "負 (停損/SL)"
                    else:
                        df_trades.at[idx, "最終結果"] = "勝 (停利/TP)"
                    return

            # 截止前沒有碰到停損或停利，結算為比賽已結束。
            if up_to_time >= COMPETITION_END:
                mark_competition_ended(idx)

    # 依序審核每位參賽者的行為軌跡
    for user in users:
        user_idx = df_trades[df_trades["參賽者"] == user].index
        active_trade_idx = None

        for idx in user_idx:
            row = df_trades.loc[idx]
            current_time = row["時間戳記"]
            action = str(row["直接進場/預掛價格/撤單"])

            # 先把上一筆有效訂單推進到「本筆時間」或「比賽截止」的較早者。
            if active_trade_idx is not None:
                update_active_trade(
                    active_trade_idx,
                    min(current_time, COMPETITION_END),
                )

                status = df_trades.at[active_trade_idx, "最終結果"]
                if status not in ["未成交 (掛單中)", "待處理", "持倉中"]:
                    active_trade_idx = None

            # 截止時間以後的任何新紀錄都不再影響比賽。
            if current_time > COMPETITION_END:
                mark_competition_ended(idx)
                continue

            # 處理撤單
            if "撤單" in action:
                df_trades.at[idx, "最終結果"] = "撤單操作紀錄"

                if active_trade_idx is not None:
                    status = str(df_trades.at[active_trade_idx, "最終結果"])

                    if "未成交" in status or "待處理" in status:
                        df_trades.at[active_trade_idx, "最終結果"] = "已主動撤銷"
                        df_trades.at[active_trade_idx, "實際離場時間"] = current_time
                        active_trade_idx = None
                continue

            # 防止重疊下單
            if active_trade_idx is not None:
                df_trades.at[idx, "最終結果"] = "無效單 (已有持倉或掛單中)"
                continue

            # 建立新單
            df_trades.at[idx, "最終結果"] = "未成交 (掛單中)"

            if "直接進場" in action:
                df_trades.at[idx, "過期時間"] = current_time + timedelta(hours=1)
            else:
                match = re.search(r"(\d+)天", action)
                days = int(match.group(1)) if match else 1
                df_trades.at[idx, "過期時間"] = current_time + timedelta(days=days)

            active_trade_idx = idx

        # 不再推演到市場最新時間，只結算到比賽截止時間。
        if active_trade_idx is not None:
            update_active_trade(active_trade_idx, COMPETITION_END)

    return df_trades


# ==========================================
# 4. 舊版獨立回測函數（保留相容性）
# ==========================================
def run_backtest(df_trades, df_kline):
    """
    舊版獨立回測函數。
    UI 使用 run_comprehensive_backtest()；此函數也限制只看到比賽截止行情。
    """
    df_trades = df_trades.copy()
    df_trades["實際進場時間"] = pd.NaT
    df_trades["實際離場時間"] = pd.NaT
    df_trades["最終結果"] = df_trades["系統狀態"]

    competition_klines = df_kline[df_kline.index <= COMPETITION_END]

    for idx, row in df_trades.iterrows():
        if row["系統狀態"] != "待處理":
            continue

        record_time = row["時間戳記"]
        if record_time > COMPETITION_END:
            df_trades.at[idx, "最終結果"] = COMPETITION_ENDED_STATUS
            df_trades.at[idx, "實際離場時間"] = COMPETITION_END
            continue

        entry_price = float(row["進場價位"])
        sl = float(row["停損"])
        tp = float(row["停利"])
        direction = str(row["多/ 空"]).strip()
        action_type = str(row["直接進場/預掛價格/撤單"])

        actual_entry = None
        actual_exit = None
        result = "未成交 (掛單中)"

        if "直接進場" in action_type:
            validation_start = record_time - timedelta(hours=1)
            validation_end = min(record_time + timedelta(hours=1), COMPETITION_END)
            validation_klines = competition_klines[
                (competition_klines.index >= validation_start)
                & (competition_klines.index <= validation_end)
            ].dropna(subset=["low", "high"])

            price_touched = any(
                candle["low"] <= entry_price <= candle["high"]
                for _, candle in validation_klines.iterrows()
            )

            if price_touched:
                actual_entry = record_time
            elif record_time + timedelta(hours=1) <= COMPETITION_END:
                result = "未成交 (直接進場前後2小時未觸價)"
            else:
                result = COMPETITION_ENDED_STATUS
                actual_exit = COMPETITION_END
        else:
            future_klines = competition_klines[
                competition_klines.index >= record_time
            ].dropna(subset=["low", "high"])

            for candle_time, candle in future_klines.iterrows():
                if candle["low"] <= entry_price <= candle["high"]:
                    actual_entry = candle_time
                    break

            if actual_entry is None:
                result = COMPETITION_ENDED_STATUS
                actual_exit = COMPETITION_END

        if actual_entry is not None:
            result = "持倉中"
            exit_klines = competition_klines[
                competition_klines.index > actual_entry
            ].dropna(subset=["low", "high"])

            for candle_time, candle in exit_klines.iterrows():
                hit_sl = False
                hit_tp = False

                if "空" in direction:
                    hit_sl = candle["high"] >= sl
                    hit_tp = candle["low"] <= tp
                elif "多" in direction:
                    hit_sl = candle["low"] <= sl
                    hit_tp = candle["high"] >= tp

                if hit_sl or hit_tp:
                    actual_exit = candle_time
                    if hit_sl and hit_tp:
                        result = "負 (插針雙殺算停損)"
                    elif hit_sl:
                        result = "負 (停損/SL)"
                    else:
                        result = "勝 (停利/TP)"
                    break

            if result == "持倉中":
                result = COMPETITION_ENDED_STATUS
                actual_exit = COMPETITION_END

        df_trades.at[idx, "實際進場時間"] = actual_entry
        df_trades.at[idx, "實際離場時間"] = actual_exit
        df_trades.at[idx, "最終結果"] = result

    return df_trades


# ==========================================
# 5. UI 與視覺化渲染
# ==========================================
st.title("🏆 ETH 模擬交易競賽戰情室")
st.caption(
    f"比賽已於 {COMPETITION_END_LABEL} 截止。"
    "截止後的新單，以及截止時仍未結案的掛單／持倉，均顯示「比賽已結束」。"
)

refresh_price = st.button("🔄 更新幣安最新價格")
if refresh_price:
    fetch_binance_latest_price.clear()
    fetch_binance_klines.clear()

latest_price, latest_price_time = fetch_binance_latest_price()

price_col, status_col = st.columns(2)
with price_col:
    if latest_price is not None:
        st.metric(
            "Binance ETHUSDT 永續｜最新成交價",
            f"{latest_price:,.2f} USDT",
        )
        st.caption(
            f"成交時間：{latest_price_time:%Y/%m/%d %H:%M:%S}（台北）｜"
            "來源：Binance USDⓈ-M Futures Last Price"
        )
    else:
        st.metric("Binance ETHUSDT 永續｜最新成交價", "讀取失敗")

with status_col:
    st.metric("比賽狀態", "已結束")
    st.caption(f"最終結算時間：{COMPETITION_END_LABEL}")

# 獲取與運算數據
df_raw = load_and_clean_data(GOOGLE_SHEET_CSV_URL)
df_kline = fetch_binance_klines()

if not df_raw.empty and not df_kline.empty:
    # 雙重防線：回測資料先切到截止時間，回測函數內也會再次限制。
    df_competition_kline = df_kline[df_kline.index <= COMPETITION_END].copy()
    df_result = run_comprehensive_backtest(
        df_raw.copy(),
        df_competition_kline,
    )

    # --- 模組 A：戰力排行榜 ---
    st.subheader("🔥 最終戰力排行榜")
    leaderboard_data = []
    users = df_result["參賽者"].dropna().unique()

    for user in users:
        user_df = df_result[df_result["參賽者"] == user]
        wins = len(user_df[user_df["最終結果"].str.contains("勝", na=False)])
        losses = len(user_df[user_df["最終結果"].str.contains("負", na=False)])
        cancel_count = len(
            user_df[user_df["最終結果"].str.contains("撤銷", na=False)]
        )
        competition_ended_count = len(
            user_df[user_df["最終結果"] == COMPETITION_ENDED_STATUS]
        )

        # 平均持倉時間只計算真正以勝／負結案的交易，不把比賽截止強制結束混入。
        finished_trades = user_df[
            user_df["實際離場時間"].notnull()
            & user_df["實際進場時間"].notnull()
            & user_df["最終結果"].str.contains("勝|負", regex=True, na=False)
        ]

        if not finished_trades.empty:
            avg_duration = (
                finished_trades["實際離場時間"]
                - finished_trades["實際進場時間"]
            ).mean()

            if pd.notnull(avg_duration):
                avg_duration_str = (
                    f"{avg_duration.components.days}天 "
                    f"{avg_duration.components.hours}小時"
                )
            else:
                avg_duration_str = "無"
        else:
            avg_duration_str = "無"

        leaderboard_data.append(
            {
                "參賽者": user,
                "淨勝分 (勝-負)": wins - losses,
                "勝 / 負": f"{wins} / {losses}",
                "比賽結束未結案／截止後紀錄": competition_ended_count,
                "主動撤單次數": cancel_count,
                "平均持倉時間": avg_duration_str,
            }
        )

    if leaderboard_data:
        df_leaderboard = (
            pd.DataFrame(leaderboard_data)
            .sort_values("淨勝分 (勝-負)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df_leaderboard, use_container_width=True)

    st.divider()

    # --- 模組 B：個人覆盤 Dashboard ---
    st.subheader("🔍 個人交易軌跡覆盤")
    if len(users) > 0:
        selected_user = st.selectbox("選擇要覆盤的參賽者：", users)
        user_trades = df_result[df_result["參賽者"] == selected_user]

        # 圖表顯示幣安成交價 K 線；可繼續顯示比賽後行情，
        # 但交易結果已固定在截止時間，不會被後續行情改寫。
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_kline.index,
                    open=df_kline["open"],
                    high=df_kline["high"],
                    low=df_kline["low"],
                    close=df_kline["close"],
                    name="Binance ETHUSDT Perpetual 1H",
                )
            ]
        )

        fig.add_vline(
            x=COMPETITION_END,
            line_dash="dash",
            line_width=2,
            annotation_text="比賽截止",
            annotation_position="top left",
        )

        for _, row in user_trades.iterrows():
            if pd.notnull(row["實際進場時間"]):
                entry_t = row["實際進場時間"]
                exit_t = (
                    row["實際離場時間"]
                    if pd.notnull(row["實際離場時間"])
                    else COMPETITION_END
                )
                entry_p = float(row["進場價位"])
                direction = row["多/ 空"]
                result = str(row["最終結果"])
                reason = row.get("進場理由", "無")

                if "勝" in result:
                    box_color = "rgba(0, 255, 0, 0.2)"
                    border_color = "green"
                elif "負" in result:
                    box_color = "rgba(255, 0, 0, 0.2)"
                    border_color = "red"
                else:
                    box_color = "rgba(255, 255, 0, 0.2)"
                    border_color = "yellow"

                y_max = entry_p * 1.05
                y_min = entry_p * 0.95
                mask = (df_kline.index >= entry_t) & (df_kline.index <= exit_t)
                price_window = df_kline.loc[mask].dropna(subset=["high", "low"])
                if not price_window.empty:
                    y_max = price_window["high"].max()
                    y_min = price_window["low"].min()

                fig.add_shape(
                    type="rect",
                    x0=entry_t,
                    y0=y_min,
                    x1=exit_t,
                    y1=y_max,
                    line=dict(color=border_color, width=1.5),
                    fillcolor=box_color,
                )

                symbol = "triangle-up" if "多" in str(direction) else "triangle-down"
                arrow_color = "green" if "多" in str(direction) else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[entry_t],
                        y=[entry_p],
                        mode="markers",
                        marker=dict(
                            symbol=symbol,
                            size=15,
                            color=arrow_color,
                            line=dict(width=1, color="white"),
                        ),
                        name=f"{direction} ({entry_p})",
                        hovertemplate=(
                            f"結果: {result}<br>"
                            f"價位: {entry_p}<br>"
                            f"理由: {reason}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # --- 模組 C：詳細交易紀錄表 ---
        st.subheader("📝 原始交易日誌")
        display_cols = [
            "時間戳記",
            "參賽者",
            "多/ 空",
            "進場價位",
            "最終結果",
            "實際進場時間",
            "實際離場時間",
            "進場理由",
        ]
        existing_cols = [column for column in display_cols if column in df_result.columns]
        st.dataframe(user_trades[existing_cols], use_container_width=True)
else:
    st.warning("請確認 Google Sheet 與幣安市場資料來源可以正常連線。")
