import datetime
import os
import sqlite3
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

# 強制鎖定工作目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import application
import db_manager

st.set_page_config(
    page_title="台中霧峰 PM2.5 未來 24 小時預測系統",
    layout="wide",
    page_icon="🌬️",
)


@st.cache_data(ttl=3600)
def dynamic_predict_24h(current_hour, live_features_list):
    """根據當前動態基準時間與即時特徵，使用 Attention LSTM 模型直接推論未來 24 小時數值"""
    feature_cols = [
        "測站氣壓(hPa)", "氣溫(℃)", "相對溼度(%)", "風速(m/s)", "風向(360degree)",
        "降水量(mm)", "pm25", "pm25_diff", "pm25_accel", "traffic_diff",
        "traffic_accel", "03F2100N", "03F2100S", "03F2125N", "03F2129S",
        "hour_sin", "hour_cos",
    ]
    target_col = "pm25"
    pm25_idx = feature_cols.index("pm25")

    csv_path = "dataset_for_lstm.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("找不到 dataset_for_lstm.csv")

    df_history = pd.read_csv(csv_path)

    df_history["dt"] = pd.to_datetime(
        df_history["time"] if "time" in df_history.columns else df_history.iloc[:, 0]
    )
    df_history["hour"] = df_history["dt"].dt.hour
    df_history["hour_sin"] = np.sin(2 * np.pi * df_history["hour"] / 24.0)
    df_history["hour_cos"] = np.cos(2 * np.pi * df_history["hour"] / 24.0)

    df_history["pm25_diff"] = df_history["pm25"].diff().fillna(0)
    df_history["pm25_accel"] = df_history["pm25_diff"].diff().fillna(0)

    traffic_sum = df_history[["03F2100N", "03F2100S", "03F2125N", "03F2129S"]].sum(axis=1)
    df_history["traffic_diff"] = traffic_sum.diff().fillna(0)
    df_history["traffic_accel"] = df_history["traffic_diff"].diff().fillna(0)

    scaler_X = StandardScaler().fit(df_history[feature_cols])
    scaler_y = StandardScaler().fit(df_history[[target_col]])

    live_features_np = np.array(live_features_list, dtype=np.float32)
    recent_23 = df_history[feature_cols].iloc[-23:].values
    rolling_window = np.vstack([recent_23, live_features_np])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = application.AttentionMultiStepLSTM(
        input_size=17, hidden_size=64, output_steps=24
    ).to(device)

    model_path = "best_lstm_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重 `{model_path}`")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    current_hour_naive = current_hour.replace(tzinfo=None)

    window_df = pd.DataFrame(rolling_window, columns=feature_cols)
    window_scaled = scaler_X.transform(window_df)

    input_tensor = (
        torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        preds_delta_scaled = model(input_tensor).cpu().numpy()[0]

    base_pm25_scaled = window_scaled[-1, pm25_idx]
    pred_y_scaled = base_pm25_scaled + preds_delta_scaled
    preds_pm25 = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).flatten()

    future_predictions = []
    future_times = []

    for step in range(1, 25):
        pred_val = max(0.0, float(preds_pm25[step - 1]))
        future_predictions.append(round(pred_val, 2))
        future_time = current_hour_naive + datetime.timedelta(hours=step)
        future_times.append(future_time)

    df_result = pd.DataFrame(
        {"target_datetime": future_times, "predicted_pm25": future_predictions}
    )
    return df_result


def get_fallback_features(prev_hour):
    """取得備援數據"""
    feature_cols = [
        "測站氣壓(hPa)", "氣溫(℃)", "相對溼度(%)", "風速(m/s)", "風向(360degree)",
        "降水量(mm)", "pm25", "pm25_diff", "pm25_accel", "traffic_diff",
        "traffic_accel", "03F2100N", "03F2100S", "03F2125N", "03F2129S",
        "hour_sin", "hour_cos",
    ]

    try:
        conn = sqlite3.connect("pm25_forecast.db")
        df_db = pd.read_sql(
            "SELECT * FROM realtime_logs ORDER BY timestamp DESC LIMIT 1", conn
        )
        conn.close()
        if not df_db.empty and all(col in df_db.columns for col in feature_cols):
            return df_db[feature_cols].iloc[0].tolist(), ["ℹ️ 成功使用 SQLite 資料庫紀錄作為車流與氣象備援數據"]
    except Exception:
        pass

    h = prev_hour.hour
    sin_h = float(np.sin(2 * np.pi * h / 24.0))
    cos_h = float(np.cos(2 * np.pi * h / 24.0))

    fallback_data = [
        1008.5, 24.5, 75.0, 1.8, 180.0, 0.0, 15.0,
        0.0, 0.0, 0.0, 0.0,
        620.0, 580.0, 510.0, 490.0,
        sin_h, cos_h
    ]
    return fallback_data, ["ℹ️ API 連線失敗，已載入動態預設車流與氣象備援數值"]


def render_backtest_section():
    """歷史追溯驗證 (Backtesting) 區塊"""
    st.markdown("---")
    st.subheader("📈 過去 24 小時歷史追溯驗證 (Backtesting)")
    st.caption("自動比對過去 24 小時『歷史預測值』與『實際觀測值』之模型表現")

    try:
        conn = sqlite3.connect("pm25_forecast.db")
        query = """
            SELECT timestamp, real_pm25, pred_pm25 
            FROM prediction_logs 
            ORDER BY timestamp DESC LIMIT 24
        """
        df_backtest = pd.read_sql(query, conn)
        conn.close()
        df_backtest = df_backtest.iloc[::-1].reset_index(drop=True)
    except Exception:
        st.info("💡 資料庫累積對照數據中，目前呈現系統自動驗證圖表...")
        return

    if df_backtest.empty:
        st.warning("⚠️ 尚未建立足夠的歷史預測與實測對照紀錄。")
        return

    y_true = df_backtest["real_pm25"].values
    y_pred = df_backtest["pred_pm25"].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("對照點數", f"{len(df_backtest)} 小時")
    col2.metric("平均絕對誤差 (MAE)", f"{mae:.2f} µg/m³")
    col3.metric("均方根誤差 (RMSE)", f"{rmse:.2f} µg/m³")
    col4.metric("平均絕對百分比誤差 (MAPE)", f"{mape:.2f} %")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_backtest["timestamp"], y=df_backtest["real_pm25"],
        mode="lines+markers", name="過去 24 小時 PM2.5 實測值",
        line=dict(color="#d62728", width=2.5), marker=dict(symbol="square", size=7)
    ))

    fig.add_trace(go.Scatter(
        x=df_backtest["timestamp"], y=df_backtest["pred_pm25"],
        mode="lines+markers", name="歷史對應時間預測值",
        line=dict(color="#1f77b4", width=2, dash="dash"), marker=dict(symbol="circle", size=6)
    ))

    fig.add_hline(
        y=15.4, line_dash="dashdot", line_color="green",
        annotation_text="AQI 良好邊界 (15.4 µg/m³)", annotation_position="top right"
    )

    fig.update_layout(
        title=f"霧峰區 PM2.5 過去 {len(df_backtest)} 小時歷史實測與預測對照圖",
        xaxis_title="時間 (DateTime)", yaxis_title="PM2.5 濃度 (µg/m³)",
        hovermode="x unified", height=450, margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🌬️ 台中市霧峰區 PM2.5 未來 24 小時預測系統")
    st.caption("結合大氣氣象、即時環測與國道 3 號車流量之 Attention LSTM 深度學習趨勢預測儀表板")

    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.cache_data.clear()
        st.rerun()

    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    current_hour = now.replace(minute=0, second=0, microsecond=0)
    prev_hour = current_hour - datetime.timedelta(hours=1)

    current_time_str = current_hour.strftime("%Y-%m-%d %H:00")
    prev_time_str = prev_hour.strftime("%H:00")
    traffic_range_str = f"{prev_time_str} ~ {current_hour.strftime('%H:00')}"

    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")
    st.sidebar.write(f"🚗 **車流統計區間**: {traffic_range_str}")

    # 擷取即時數據
    live_features_list = []
    fetch_logs = []
    with st.spinner(f"📡 正在擷取霧峰即時監測與車流數據 ({traffic_range_str})..."):
        try:
            csv_path = "dataset_for_lstm.csv"
            df_history = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

            live_features, fetch_logs = application.fetch_wufeng_live_features(df_history)
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            st.warning(f"⚠️ 即時 API 擷取異常 ({e})，已切換至 [{traffic_range_str}] 備援數據。")
            live_features_list, fetch_logs = get_fallback_features(prev_hour)

    # 4 個門架車流量
    g_2100N = live_features_list[11]
    g_2100S = live_features_list[12]
    g_2125N = live_features_list[13]
    g_2129S = live_features_list[14]

    traffic_north = g_2100N + g_2125N
    traffic_south = g_2100S + g_2129S
    traffic_total = traffic_north + traffic_south

    # 數據 Summary
    st.subheader(f"📊 即時監測與車流 Summary ({traffic_range_str} 累積值)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("即時 PM2.5", f"{live_features_list[6]:.1f} µg/m³")
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")

    st.markdown("##### 🚗 國道 3 號霧峰段車流量數據")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("北上總車流量 (2100N + 2125N)", f"{int(traffic_north):,} 輛")
    col_t2.metric("南下總車流量 (2100S + 2129S)", f"{int(traffic_south):,} 輛")
    col_t3.metric("🔥 雙向全區總車流量", f"{int(traffic_total):,} 輛")

    # ------------------------------------------------------------------
    # 📝 核心工作日誌 (顯示下載車流檔名、門架細節與調試歷程)
    # ------------------------------------------------------------------
    with st.expander("📝 查看車流量下載檔名、門架細節與完整工作日誌 (Work Log)", expanded=True):
        st.markdown(f"**⏰ 執行時間點**：`{current_time_str}` | **車流統計時段**：`{traffic_range_str}`")

        # 門架流量明細表格
        df_traffic_log = pd.DataFrame({
            "門架代碼": ["03F2100N (北上)", "03F2100S (南下)", "03F2125N (北上)", "03F2129S (南下)", "全區車流總和"],
            "對應門架位置/方向": ["國3 210k+000 北上", "國3 210k+000 南下", "國3 212k+500 北上", "國3 212k+900 南下", "霧峰段雙向 4 門架加總"],
            "門架總車流量 (輛)": [f"{int(g_2100N):,}", f"{int(g_2100S):,}", f"{int(g_2125N):,}", f"{int(g_2129S):,}", f"👉 {int(traffic_total):,} 輛"]
        })
        st.table(df_traffic_log)

        # 完整爬蟲調試歷程紀錄 (Terminal 日誌完全呈現在 Streamlit)
        st.markdown("**🔍 完整高公局 CSV 與環測 API 調試歷程 (Debug Console)：**")
        if fetch_logs:
            full_log_str = "\n".join(fetch_logs)
            st.code(full_log_str, language="text")
        else:
            st.info("尚無調試歷程訊息，請按左側「🔄 刷新即時監測數據」")

    st.markdown("---")
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    # 推論未來 24 小時
    with st.spinner("🔮 正在根據最新基準時間即時計算未來 24 小時趨勢..."):
        try:
            df_pred = dynamic_predict_24h(current_hour, live_features_list)
        except Exception as e:
            st.error(f"❌ 模型推論發生錯誤: {e}")
            return

    current_hour_naive = current_hour.replace(tzinfo=None)
    end_time_24h = current_hour_naive + datetime.timedelta(hours=24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[current_hour_naive], y=[live_features_list[6]],
        mode="markers", name="當前實測值 (基準時間)",
        marker=dict(color="red", size=12),
    ))

    fig.add_trace(go.Scatter(
        x=df_pred["target_datetime"], y=df_pred["predicted_pm25"],
        mode="lines+markers", name="LSTM 預測 PM2.5 (µg/m³)",
        line=dict(color="#0083B0", width=3), marker=dict(size=6),
    ))

    fig.add_hline(y=15, line_dash="dash", line_color="orange", annotation_text="WHO 24小時建議值 (15 µg/m³)")
    fig.add_hline(y=35.5, line_dash="dash", line_color="red", annotation_text="環境部橘色提醒臨界點 (35.5 µg/m³)")

    fig.update_layout(
        xaxis=dict(title="預測時間點", type="date", tickformat="%m/%d %H:00", range=[current_hour_naive, end_time_24h]),
        yaxis_title="PM2.5 濃度 (µg/m³)", hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20), height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 預測數值明細
    st.subheader("📋 未來 24 小時預測數值明細")
    df_display = pd.DataFrame({
        "預測時間點": df_pred["target_datetime"].dt.strftime("%m/%d %H:00"),
        "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
    })
    st.dataframe(df_display.astype(str), use_container_width=True)

    # 歷史追溯驗證
    render_backtest_section()


if __name__ == "__main__":
    main()
