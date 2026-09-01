import datetime
import os
import re
import sys
import urllib3
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# 引入 DB 操作模組
import db_manager

# 關閉 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API 金鑰設定
CWA_API_KEY = "CWA-F6B5F348-77D8-4EA8-8874-FBA50E6191DE"
MOENV_API_KEY = "5ae4f1a2-b6e6-4b79-82c8-0c84d694b7a7"


# 1. 定義 Direct Multi-Step LSTM + Self-Attention 模型架構
class AttentionMultiStepLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=2, output_steps=24):
        super(AttentionMultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_steps)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 2. 自動化擷取【霧峰區】即時 17 項特徵 (含偵錯 Log 回傳)
def fetch_wufeng_live_features(df_history=None):
    debug_logs = []
    now = datetime.datetime.now()
    debug_logs.append(f"🔍 [Debug] 開始執行即時數據擷取任務，系統時間: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    # (A) 霧峰 PM2.5
    pm25 = None
    try:
        url_epb_table = "https://taqm.epb.taichung.gov.tw/TQAMPM25table.ASPX"
        res_epb = requests.get(
            url_epb_table, headers=headers, timeout=10, verify=False
        )
        res_epb.encoding = "utf-8"
        soup = BeautifulSoup(res_epb.text, "html.parser")
        all_cells = [
            tag.text.strip() for tag in soup.find_all(["td", "th", "a"])
        ]

        for idx, text in enumerate(all_cells):
            if "霧峰" in text and idx + 1 < len(all_cells):
                val_str = all_cells[idx + 1]
                if val_str.isdigit() or re.match(r"^\d+(\.\d+)?$", val_str):
                    pm25 = float(val_str)
                    debug_logs.append(
                        f"   [1/3] ✅ 【臺中環保局】霧峰站即時 PM2.5 成功解析: {pm25} µg/m³"
                    )
                    break
    except Exception as e:
        debug_logs.append(f"   [1/3] ℹ️ 臺中環保局網頁爬取跳過/失敗: {e}")

    if pm25 is None:
        try:
            url_dali = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=5&format=json&filters=sitename,eq,大里"
            res_dali = requests.get(
                url_dali, headers=headers, timeout=10, verify=False
            ).json()
            recs = (
                res_dali.get("records", [])
                if isinstance(res_dali, dict)
                else res_dali
            )
            if recs:
                val = recs[0].get("pm25") or recs[0].get("pm2.5")
                if val:
                    pm25 = float(val)
                    debug_logs.append(
                        f"   [1/3] ✅ 採用鄰近【大里標準站】即時 PM2.5: {pm25} µg/m³"
                    )
        except Exception as e:
            debug_logs.append(f"   [1/3] ℹ️ 大里站 API 跳過: {e}")

    if pm25 is None:
        pm25 = (
            float(df_history["pm25"].iloc[-1])
            if df_history is not None
            else 15.0
        )
        debug_logs.append(f"   [1/3] ⚠️ PM2.5 採用歷史/保底數值: {pm25} µg/m³")

    # (B) 霧峰氣象
    press, temp, rh, wind_spd, wind_dir, rain = 995.0, 25.0, 75.0, 1.5, 180.0, 0.0
    try:
        url_cwa = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_API_KEY}&StationName=霧峰"
        res_cwa = requests.get(
            url_cwa, headers=headers, timeout=10, verify=False
        ).json()
        if (
            isinstance(res_cwa, dict)
            and res_cwa.get("records")
            and res_cwa["records"].get("Station")
        ):
            station_data = res_cwa["records"]["Station"][0]
            obs_time_str = station_data.get("ObsTime", {}).get(
                "DateTime", "未知時間"
            )
            station_elem = station_data["WeatherElement"]

            def safe_float(val, default_val):
                try:
                    v = float(val)
                    return default_val if v < -90 else v
                except (ValueError, TypeError):
                    return default_val

            press = safe_float(station_elem.get("AirPressure"), press)
            temp = safe_float(station_elem.get("AirTemperature"), temp)
            rh = safe_float(station_elem.get("RelativeHumidity"), rh)
            wind_spd = safe_float(station_elem.get("WindSpeed"), wind_spd)
            wind_dir = safe_float(station_elem.get("WindDirection"), wind_dir)
            if "Now" in station_elem and isinstance(station_elem["Now"], dict):
                rain = safe_float(station_elem["Now"].get("Precipitation"), 0.0)

            debug_logs.append(
                f"   [2/3] ✅ 成功取得【氣象署霧峰站】 (時間: {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%, 氣壓 {press}hPa"
            )
    except Exception as e:
        debug_logs.append(f"   [2/3] ⚠️ 氣象署 API 解析失敗，採用保底數值: {e}")

    # (C) 國道 3 號車流量 (含完整除錯診斷與全車種累加)
    v_2100N, v_2100S, v_2125N, v_2129S = 0.0, 0.0, 0.0, 0.0
    WORKER_PROXY = "https://steep-wood-cf94.4b432104.workers.dev/?url="

    try:
        traffic_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        latest_base_time = None

        debug_logs.append("🚗 [車流探測] 開始向高公局伺服器探測最新 5 分鐘 CSV 車流檔案...")

        # 往前推算 10 ~ 40 分鐘，尋找高公局伺服器上已產生的 CSV
        for offset_min in range(10, 45, 5):
            probe_time = now - datetime.timedelta(minutes=offset_min)
            check_time = probe_time.replace(
                minute=(probe_time.minute // 5) * 5, second=0, microsecond=0
            )
            ymd = check_time.strftime("%Y%m%d")
            hh = check_time.strftime("%H")
            mm = check_time.strftime("%M")

            raw_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"
            proxy_url = f"{WORKER_PROXY}{raw_url}"

            debug_logs.append(f"   📡 測試時間點 {ymd} {hh}:{mm} ...")

            for mode, try_url in [("直連", raw_url), ("Worker代理", proxy_url)]:
                try:
                    res_check = requests.get(
                        try_url, headers=traffic_headers, timeout=4, verify=False
                    )
                    debug_logs.append(
                        f"      -> [{mode}] HTTP Response Status: {res_check.status_code}, Body Length: {len(res_check.text)} bytes"
                    )
                    if res_check.status_code == 200 and len(res_check.text) > 100:
                        latest_base_time = check_time
                        debug_logs.append(
                            f"   🎯 成功定位高公局最新車流 CSV 時間點: {ymd} {hh}:{mm} (管道: {mode})"
                        )
                        break
                except Exception as ex:
                    debug_logs.append(f"      ❌ [{mode}] 連線異常: {ex}")

            if latest_base_time:
                break

        if latest_base_time:
            hourly_vol = {
                "03F2100N": 0.0,
                "03F2100S": 0.0,
                "03F2125N": 0.0,
                "03F2129S": 0.0,
            }
            success_count = 0
            total_lines_scanned = 0

            for i in range(11, -1, -1):
                target_time = latest_base_time - datetime.timedelta(minutes=i * 5)
                ymd = target_time.strftime("%Y%m%d")
                hh = target_time.strftime("%H")
                mm = target_time.strftime("%M")

                raw_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"
                proxy_url = f"{WORKER_PROXY}{raw_url}"

                res_csv = None
                for try_url in [raw_url, proxy_url]:
                    try:
                        resp = requests.get(
                            try_url, headers=traffic_headers, timeout=4, verify=False
                        )
                        if resp.status_code == 200 and len(resp.text) > 100:
                            res_csv = resp
                            break
                    except Exception:
                        continue

                if res_csv:
                    success_count += 1
                    lines = res_csv.text.strip().split("\n")
                    total_lines_scanned += len(lines)
                    matched_in_file = 0

                    for line in lines:
                        parts = line.split(",")
                        if len(parts) >= 5:
                            gantry_id = parts[1].strip()
                            vehicle_cnt = parts[4].strip()
                            if gantry_id in hourly_vol:
                                try:
                                    # 累加所有車種流量
                                    hourly_vol[gantry_id] += float(vehicle_cnt)
                                    matched_in_file += 1
                                except ValueError:
                                    pass
                    debug_logs.append(
                        f"   📄 [CSV {hh}:{mm}] 讀取 {len(lines)} 行，匹配霧峰段門架 {matched_in_file} 次"
                    )

            debug_logs.append(
                f"📊 累計下載成功 {success_count}/12 份 CSV 檔，共掃描 {total_lines_scanned} 行資料。"
            )
            debug_logs.append(f"🔢 各門架原始實測總量 (車輛數): {hourly_vol}")

            if success_count > 0:
                scale_factor = 12.0 / success_count
                v_2100N = hourly_vol["03F2100N"] * scale_factor
                v_2100S = hourly_vol["03F2100S"] * scale_factor
                v_2125N = hourly_vol["03F2125N"] * scale_factor
                v_2129S = hourly_vol["03F2129S"] * scale_factor
                debug_logs.append(
                    f"   [3/3] ✅ 車流計算完成 (等比換算一小時): 2100N={v_2100N:.0f}, 2100S={v_2100S:.0f}, 2125N={v_2125N:.0f}, 2129S={v_2129S:.0f}"
                )
            else:
                v_2100N, v_2100S, v_2125N, v_2129S = 2200.0, 2400.0, 1800.0, 1700.0
                debug_logs.append("⚠️ 車輛 CSV 解析結果為空，使用白天真實保底流量數值")
        else:
            v_2100N, v_2100S, v_2125N, v_2129S = 2200.0, 2400.0, 1800.0, 1700.0
            debug_logs.append("❌ 無法抓取高公局 CSV 檔，採用保底預設車流量")

    except Exception as e:
        v_2100N, v_2100S, v_2125N, v_2129S = 2200.0, 2400.0, 1800.0, 1700.0
        debug_logs.append(f"   [3/3] ℹ️ 車流抓取過程發生例外: {e}，切換保底數據")

    # (D) 動態計算一階差分與二階加速度
    last_pm25 = (
        float(df_history["pm25"].iloc[-1])
        if df_history is not None
        else pm25
    )
    last_pm25_diff = (
        float(df_history["pm25_diff"].iloc[-1])
        if df_history is not None
        else 0.0
    )
    pm25_diff = pm25 - last_pm25
    pm25_accel = pm25_diff - last_pm25_diff

    current_traffic = v_2100N + v_2100S + v_2125N + v_2129S
    last_traffic = (
        float(
            df_history[["03F2100N", "03F2100S", "03F2125N", "03F2129S"]]
            .iloc[-1]
            .sum()
        )
        if df_history is not None
        else current_traffic
    )
    last_traffic_diff = (
        float(df_history["traffic_diff"].iloc[-1])
        if df_history is not None
        else 0.0
    )
    traffic_diff = current_traffic - last_traffic
    traffic_accel = traffic_diff - last_traffic_diff

    hour_sin = np.sin(2 * np.pi * now.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * now.hour / 24.0)

    features = [
        press,
        temp,
        rh,
        wind_spd,
        wind_dir,
        rain,
        pm25,
        pm25_diff,
        pm25_accel,
        traffic_diff,
        traffic_accel,
        v_2100N,
        v_2100S,
        v_2125N,
        v_2129S,
        hour_sin,
        hour_cos,
    ]

    return features, debug_logs


# 3. 主推論程式 (供獨立 CLI 執行使用)
def main():
    print("==================================================")
    print("🚀 啟動【霧峰 PM2.5 未來 24 小時 Attention 多步預測系統】")
    print("==================================================")

    db_manager.init_db()
    df_history = pd.read_csv("dataset_for_lstm.csv")

    df_history["dt"] = pd.to_datetime(
        df_history["time"]
        if "time" in df_history.columns
        else df_history.iloc[:, 0]
    )
    df_history["hour"] = df_history["dt"].dt.hour
    df_history["hour_sin"] = np.sin(2 * np.pi * df_history["hour"] / 24.0)
    df_history["hour_cos"] = np.cos(2 * np.pi * df_history["hour"] / 24.0)

    df_history["pm25_diff"] = df_history["pm25"].diff().fillna(0)
    df_history["pm25_accel"] = df_history["pm25_diff"].diff().fillna(0)

    traffic_sum = df_history[
        ["03F2100N", "03F2100S", "03F2125N", "03F2129S"]
    ].sum(axis=1)
    df_history["traffic_diff"] = traffic_sum.diff().fillna(0)
    df_history["traffic_accel"] = df_history["traffic_diff"].diff().fillna(0)

    feature_cols = [
        "測站氣壓(hPa)",
        "氣溫(℃)",
        "相對溼度(%)",
        "風速(m/s)",
        "風向(360degree)",
        "降水量(mm)",
        "pm25",
        "pm25_diff",
        "pm25_accel",
        "traffic_diff",
        "traffic_accel",
        "03F2100N",
        "03F2100S",
        "03F2125N",
        "03F2129S",
        "hour_sin",
        "hour_cos",
    ]
    target_col = "pm25"

    scaler_X = StandardScaler().fit(df_history[feature_cols])
    scaler_y = StandardScaler().fit(df_history[[target_col]])

    live_features, logs = fetch_wufeng_live_features(df_history)
    for log in logs:
        print(log)

    live_features_list = [float(x) for x in live_features]
    now = datetime.datetime.now()
    base_time = now.replace(minute=0, second=0, microsecond=0)
    current_time_str = base_time.strftime("%Y-%m-%d %H:00")

    db_manager.save_real_data(current_time_str, live_features_list)

    recent_23 = df_history[feature_cols].iloc[-23:].values
    current_window = np.vstack(
        [recent_23, np.array(live_features_list, dtype=np.float32)]
    )

    window_scaled = scaler_X.transform(
        pd.DataFrame(current_window, columns=feature_cols)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = (
        torch.tensor(window_scaled, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    model = AttentionMultiStepLSTM(
        input_size=17, hidden_size=64, output_steps=24
    ).to(device)

    model_path = "best_lstm_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds_delta_scaled = model(input_tensor).cpu().numpy()[0]

        base_pm25_scaled = window_scaled[-1, 6]
        pred_y_scaled = base_pm25_scaled + preds_delta_scaled
        preds_pm25 = scaler_y.inverse_transform(
            pred_y_scaled.reshape(-1, 1)
        ).flatten()

        predictions_to_db = []
        for i, pred in enumerate(preds_pm25):
            future_time = base_time + datetime.timedelta(hours=i + 1)
            target_time_str = future_time.strftime("%Y-%m-%d %H:00")
            pred_val = max(0.0, float(pred))
            predictions_to_db.append((target_time_str, pred_val, i + 1))

        db_manager.save_predictions(current_time_str, predictions_to_db)
        print("💾 已成功將預測數值寫入 SQLite 資料庫")


if __name__ == "__main__":
    main()
