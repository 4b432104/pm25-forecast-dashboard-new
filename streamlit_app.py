import datetime
import os
import sqlite3
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


@st.cache_data(ttl=3600)  # 快取 1 小時
def dynamic_predict_24h(current_hour, live_features_list):
    """使用最新的 Attention-LSTM (17特徵 Direct Multi-Step) 直接推論未來 24 小時數值"""
    feature_cols = [
        "測站氣壓(hPa)", "氣溫(℃)", "相對溼度(%)", "風速(m/s)", "風向(360degree)", "降水量(mm)",
        "pm25", "pm25_diff", "pm25_accel", "traffic_diff", "traffic_accel",
        "03F2100N", "03F2100S", "03F2125N", "03F2129S", "hour_sin", "hour_cos"
    ]
    target_col = "pm25"

    csv_path = "dataset_for_lstm.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("找不到 dataset_for_lstm.csv")

    df_history = pd.read_csv(csv_path)
    df_history["dt"] = pd.to_datetime(df_history["time"] if "time" in df_history.columns else df_history.iloc[:, 0])
    df_history["hour"] = df_history["dt"].dt.hour
    df_history["hour_sin"] = np.sin(2 * np.pi * df_history["hour"] / 24.0)
    df_history["hour_cos"] = np.cos(2 * np.pi * df_history["hour"] / 24.0)

    df_history["pm25_diff"] = df_history["pm25"].diff().fillna(0)
    df_history["pm25_accel"] = df_history["pm25_diff"].diff().fillna(0)

    traffic_sum = df_history[["03F2100N", "03F2100S", "03F2125N", "03F2129S"]].sum(axis=1)
    df_history["traffic_diff"] = traffic_sum.diff().fillna(0)
    df_history["traffic_accel"] = df_history["traffic_diff"].diff().fillna(0)

    # 🔧 改用 StandardScaler，對齊最新的 application.py
    scaler_X = StandardScaler().fit(df_history[feature_cols])
    scaler_y = StandardScaler().fit(df_history[[target_col]])

    # 準備模型輸入視窗 (過去 23 小時 + 當前即時值)
    live_features_np = np.array(live_features_list, dtype=np.float32)
    recent_23 = df_history[feature_cols].iloc[-23:].values
    current_window = np.vstack([recent_23, live_features_np])

    window_scaled = scaler_X.transform(pd.DataFrame(current_window, columns=feature_cols))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = application.AttentionMultiStepLSTM(input_size=17, hidden_size=64, output_steps=24).to(device)

    model_path = "best_lstm_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重 `{model_path}`")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds_delta_scaled = model(input_tensor).cpu().numpy()[0]

    # 殘差加回與反還原邏輯
    base_pm25_scaled = window_scaled[-1, 6]  # 特徵第 6 欄位是 pm25
    pred_y_scaled = base_pm25_scaled + preds_delta_scaled
    preds_pm25 = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).flatten()

    future_times = []
    future_predictions = []
    current_hour_naive = current_hour.replace(tzinfo=None)

    for i, pred in enumerate(preds_pm25):
        future_time = current_hour_naive + datetime.timedelta(hours=i + 1)
        future_times.append(future_time)
        future_predictions.append(round(max(0.0, float(pred)), 2))

    df_result = pd.DataFrame(
        {"target_datetime": future_times, "predicted_pm25": future_predictions}
    )
    return df_result


def get_fallback_features(prev_hour):
    """取得 17 項備援特徵數據"""
    try:
        conn = sqlite3.connect("prediction.db")
        df_db = pd.read_sql("SELECT * FROM real_data ORDER BY timestamp DESC LIMIT 1", conn)
        conn.close()
        if not df_db.empty:
            # 若資料庫有紀錄，取最新的 feature 陣列
            return [
                df_db['press'].iloc[0], df_db['temp'].iloc[0], df_db['rh'].iloc[0], 
                df_db['wind_spd'].iloc[0], df_db['wind_dir'].iloc[0], df_db['rain'].iloc[0],
                df_db['pm25'].iloc[0], 0.0, 0.0, 0.0, 0.0,
                df_db['v_2100N'].iloc[0], df_db['v_2100S'].iloc[0], 
                df_db['v_2125N'].iloc[0], df_db['v_2129S'].iloc[0],
                np.sin(2 * np.pi * prev_hour.hour / 24.0), np.cos(2 * np.pi * prev_hour.hour / 24.0)
            ]
    except Exception:
        pass

    # 預設保底 17 項特徵
    h = prev_hour.hour
    return [
        995.0, 25.0, 75.0, 1.5, 180.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        450.0, 480.0, 320.0, 310.0,
        float(np.sin(2 * np.pi * h / 24.0)), float(np.cos(2 * np.pi * h / 24.0))
    ]


def main():
    st.title("🌬️ 台中市霧峰區 PM2.5 未來 24 小時預測系統")
    st.caption("結合大氣氣象、即時環測與國道 3 號車流量之 Attention-LSTM 深度學習趨勢預測儀表板")

    # 初始化 DB
    db_manager.init_db()

    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 基準時間與區間
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    current_hour = now.replace(minute=0, second=0, microsecond=0)
    prev_hour = current_hour - datetime.timedelta(hours=1)

    current_time_str = current_hour.strftime("%Y-%m-%d %H:00")
    prev_time_str = prev_hour.strftime("%H:00")
    traffic_range_str = f"{prev_time_str} ~ {current_hour.strftime('%H:00')}"

    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")
    st.sidebar.write(f"🚗 **車流統計區間**: {traffic_range_str}")

    # 2. 擷取 17 項即時數據
    live_features_list = [0.0] * 17
    with st.spinner(f"📡 正在擷取霧峰即時監測與車流數據 ({traffic_range_str})..."):
        try:
            live_features = application.fetch_wufeng_live_features()
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            st.warning(f"⚠️ 即時 API 暫時無回應，切換至 [{traffic_range_str}] 動態備援數據")
            live_features_list = get_fallback_features(prev_hour)

    # 提取車流量資訊 (索引位置對齊 17 特徵)
    traffic_2100N = live_features_list[11]
    traffic_2100S = live_features_list[12]
    traffic_total = traffic_2100N + traffic_2100S + live_features_list[13] + live_features_list[14]

    # 3. 即時數據 Summary
    st.subheader(f"📊 即時監測與車流 Summary ({traffic_range_str} 累積值)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("即時 PM2.5", f"{live_features_list[6]:.1f} µg/m³")  # 索引 6 是 PM2.5
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")

    st.markdown("##### 🚗 國道 3 號霧峰段車流量統計")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("國道 3 號 (2100N 北上)", f"{int(traffic_2100N):,} 輛")
    col_t2.metric("國道 3 號 (2100S 南下)", f"{int(traffic_2100S):,} 輛")
    col_t3.metric("霧峰段 4 門架總車流量", f"{int(traffic_total):,} 輛")

    st.markdown("---")
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    # 4. 推論未來 24 小時
    with st.spinner("🔮 正在根據最新基準時間即時計算未來 24 小時趨勢..."):
        try:
            df_pred = dynamic_predict_24h(current_hour, live_features_list)
            # 自動儲存至資料庫
            predictions_to_db = [
                (row['target_datetime'].strftime("%Y-%m-%d %H:00"), row['predicted_pm25'], idx + 1)
                for idx, row in df_pred.iterrows()
            ]
            db_manager.save_predictions(current_time_str, predictions_to_db)
            db_manager.save_real_data(current_time_str, live_features_list)
        except Exception as e:
            st.error(f"❌ 模型推論發生錯誤: {e}")
            return

    # 5. 繪製圖表
    current_hour_naive = current_hour.replace(tzinfo=None)
    end_time_24h = current_hour_naive + datetime.timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[current_hour_naive],
            y=[live_features_list[6]],
            mode="markers",
            name="當前實測值 (基準時間)",
            marker=dict(color="red", size=12),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_pred["target_datetime"],
            y=df_pred["predicted_pm25"],
            mode="lines+markers",
            name="Attention-LSTM 預測 PM2.5 (µg/m³)",
            line=dict(color="#0083B0", width=3),
            marker=dict(size=6),
        )
    )

    fig.add_hline(
        y=15.4,
        line_dash="dash",
        line_color="green",
        annotation_text="AQI 良好邊界 (15.4 µg/m³)",
    )
    fig.add_hline(
        y=35.5,
        line_dash="dash",
        line_color="red",
        annotation_text="環境部橘色提醒臨界點 (35.5 µg/m³)",
    )

    fig.update_layout(
        xaxis=dict(
            title="預測時間點",
            type="date",
            tickformat="%m/%d %H:00",
            range=[current_hour_naive, end_time_24h],
        ),
        yaxis_title="PM2.5 濃度 (µg/m³)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. 明細表格
    st.subheader("📋 未來 24 小時預測數值明細")
    df_display = pd.DataFrame(
        {
            "預測時間點": df_pred["target_datetime"].dt.strftime("%m/%d %H:00"),
            "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
        }
    )
    st.dataframe(df_display.T, use_container_width=True)


if __name__ == "__main__":
    main()