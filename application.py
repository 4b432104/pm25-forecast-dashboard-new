import datetime
import io
import os
import re
import sys
import urllib3
from zoneinfo import ZoneInfo
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

# Cloudflare Proxy 轉接頭網址
CF_PROXY_URL = "https://steep-wood-cf94.4b432104.workers.dev"


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


def fetch_m03a_traffic_from_freeway():
    target_gantry = ["03F2100N", "03F2100S", "03F2125N", "03F2129S"]
    traffic_dict = {g: 0.0 for g in target_gantry}
    debug_logs = []

    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    debug_logs.append("🚗 [車流探測] 開始向高公局伺服器探測最新 5 分鐘 CSV 車流檔案...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    session = requests.Session()
    session.headers.update(headers)

    latest_valid_dt = None
    latest_channel = None

    # 優先測試 Worker 代理，再測試直連
    for minutes_back in range(10, 45, 5):
        test_dt = now - datetime.timedelta(minutes=minutes_back)
        test_dt = test_dt.replace(minute=(test_dt.minute // 5) * 5, second=0, microsecond=0)
        
        ymd_str = test_dt.strftime("%Y%m%d")
        hh_str = test_dt.strftime("%H")
        mm_str = test_dt.strftime("%M")
        
        debug_logs.append(f"📡 測試時間點 {ymd_str} {hh_str}:{mm_str} ...")
        
        raw_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd_str}/{hh_str}/TDCS_M03A_{ymd_str}_{hh_str}{mm_str}00.csv"
        proxy_url = f"{CF_PROXY_URL}/?url={raw_url}"

        # 嘗試 Worker 代理
        try:
            r_proxy = session.get(proxy_url, timeout=6, verify=False)
            if r_proxy.status_code == 200 and len(r_proxy.text) > 300:
                debug_logs.append(f"-> [Worker代理] Response 200 OK, Body Length: {len(r_proxy.text)} bytes")
                latest_valid_dt = test_dt
                latest_channel = "Worker代理"
                break
            else:
                debug_logs.append(f"❌ [Worker代理] HTTP 狀態碼異常: {r_proxy.status_code}")
        except Exception as e:
            debug_logs.append(f"❌ [Worker代理] 連線逾時/失敗: {e}")

        # 嘗試直連
        try:
            r_direct = session.get(raw_url, timeout=3, verify=False)
            if r_direct.status_code == 200 and len(r_direct.text) > 300:
                debug_logs.append(f"-> [直連] Response 200 OK, Body Length: {len(r_direct.text)} bytes")
                latest_valid_dt = test_dt
                latest_channel = "直連"
                break
        except Exception:
            debug_logs.append(f"❌ [直連] 連線 Timeout")

    # 若全數探測失敗，預設回溯 25 分鐘，並強制使用 Worker 代理
    if latest_valid_dt is None:
        latest_valid_dt = now - datetime.timedelta(minutes=25)
        latest_valid_dt = latest_valid_dt.replace(minute=(latest_valid_dt.minute // 5) * 5, second=0, microsecond=0)
        latest_channel = "Worker代理"
        debug_logs.append("⚠️ 所有即時探測嘗試均失敗，強制切換至 [Worker代理] 回溯抓取機制。")

    debug_logs.append(f"🎯 成功定位高公局最新車流 CSV 時間點: {latest_valid_dt.strftime('%Y%m%d %H:%M')} (管道: {latest_channel})")

    success_count = 0
    total_rows_scanned = 0

    # 依定位點抓取前 1 小時 (12 個 5 分鐘區間) 資料
    for i in range(11, -1, -1):
        slot_dt = latest_valid_dt - datetime.timedelta(minutes=i * 5)
        ymd = slot_dt.strftime("%Y%m%d")
        hh = slot_dt.strftime("%H")
        mm = slot_dt.strftime("%M")

        raw_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"
        target_url = f"{CF_PROXY_URL}/?url={raw_url}" if latest_channel == "Worker代理" else raw_url

        # 加入最多 2 次重試機制，且 timeout 加長至 12 秒
        for retry in range(2):
            try:
                resp = session.get(target_url, timeout=12, verify=False)
                if resp.status_code == 200 and len(resp.text) > 100:
                    csv_data = io.StringIO(resp.text)
                    df_temp = pd.read_csv(csv_data, header=None, dtype=str)

                    if len(df_temp.columns) >= 5:
                        df_temp[1] = df_temp[1].str.strip()
                        df_temp[4] = pd.to_numeric(df_temp[4], errors="coerce").fillna(0)

                        row_count = len(df_temp)
                        total_rows_scanned += row_count
                        match_count = 0

                        for gantry in target_gantry:
                            sub_df = df_temp[df_temp[1] == gantry]
                            match_count += len(sub_df)
                            vol_sum = sub_df[4].sum()
                            traffic_dict[gantry] += float(vol_sum)

                        success_count += 1
                        debug_logs.append(f"📄 [CSV {hh}:{mm}] 讀取 {row_count} 行，匹配霧峰段門架 {match_count} 次")
                        break
                else:
                    if retry == 1:
                        debug_logs.append(f"📄 [CSV {hh}:{mm}] 下載失敗 HTTP {resp.status_code}")
            except Exception as e:
                if retry == 1:
                    debug_logs.append(f"📄 [CSV {hh}:{mm}] 讀取異常: {e}")

    debug_logs.append(f"📊 累計下載成功 {success_count}/12 份 CSV 檔，共掃描 {total_rows_scanned} 行資料。")

    # 若有成功抓取資料，等比放大還原為單小時車流量；若完全失敗，填入歷史預設基準值
    if success_count > 0:
        scale_factor = 12.0 / success_count
        for gantry in target_gantry:
            traffic_dict[gantry] = round(traffic_dict[gantry] * scale_factor)
    else:
        debug_logs.append("⚠️ 車流 CSV 讀取成功率為 0，載入預設保底車流數值。")
        traffic_dict = {"03F2100N": 620.0, "03F2100S": 580.0, "03F2125N": 510.0, "03F2129S": 490.0}

    debug_logs.append(f"🔢 各門架計算總量 (輛/小時): {traffic_dict}")
    return traffic_dict, debug_logs


def fetch_wufeng_live_features(df_history=None):
    debug_logs = []
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)
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
        res_epb = requests.get(url_epb_table, headers=headers, timeout=10, verify=False)
        res_epb.encoding = "utf-8"
        soup = BeautifulSoup(res_epb.text, "html.parser")
        all_cells = [tag.text.strip() for tag in soup.find_all(["td", "th", "a"])]

        for idx, text in enumerate(all_cells):
            if "霧峰" in text and idx + 1 < len(all_cells):
                val_str = all_cells[idx + 1]
                if val_str.isdigit() or re.match(r"^\d+(\.\d+)?$", val_str):
                    pm25 = float(val_str)
                    debug_logs.append(f"[1/3] ✅ 【臺中環保局】霧峰站即時 PM2.5 成功解析: {pm25} µg/m³")
                    break
    except Exception:
        pass

    if pm25 is None:
        try:
            url_dali = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=5&format=json&filters=sitename,eq,大里"
            res_dali = requests.get(url_dali, headers=headers, timeout=10, verify=False).json()
            recs = res_dali.get("records", []) if isinstance(res_dali, dict) else res_dali
            if recs:
                val = recs[0].get("pm25") or recs[0].get("pm2.5")
                if val:
                    pm25 = float(val)
                    debug_logs.append(f"[1/3] ✅ 採用鄰近【大里標準站】即時 PM2.5: {pm25} µg/m³")
        except Exception:
            pass

    if pm25 is None:
        pm25 = float(df_history["pm25"].iloc[-1]) if (df_history is not None and "pm25" in df_history.columns) else 15.0
        debug_logs.append(f"[1/3] ⚠️ PM2.5 採用歷史/保底數值: {pm25} µg/m³")

    # (B) 霧峰氣象
    press, temp, rh, wind_spd, wind_dir, rain = 1008.5, 26.8, 91.0, 1.8, 180.0, 0.0
    try:
        url_cwa = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_API_KEY}&StationName=霧峰"
        res_cwa = requests.get(url_cwa, headers=headers, timeout=10, verify=False).json()
        if isinstance(res_cwa, dict) and res_cwa.get("records") and res_cwa["records"].get("Station"):
            station_data = res_cwa["records"]["Station"][0]
            obs_time_str = station_data.get("ObsTime", {}).get("DateTime", f"{now.strftime('%Y-%m-%d')}T09:00:00+08:00")
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
                f"[2/3] ✅ 成功取得【氣象署霧峰站】 (時間: {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%, 氣壓 {press}hPa"
            )
    except Exception:
        obs_time_str = f"{now.strftime('%Y-%m-%d')}T09:00:00+08:00"
        debug_logs.append(f"[2/3] ⚠️ 採用氣象保底數據 (時間: {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%, 氣壓 {press}hPa")

    # (C) 車流量
    traffic_dict, traffic_logs = fetch_m03a_traffic_from_freeway()
    debug_logs.extend(traffic_logs)

    v_2100N = traffic_dict.get("03F2100N", 620.0)
    v_2100S = traffic_dict.get("03F2100S", 580.0)
    v_2125N = traffic_dict.get("03F2125N", 510.0)
    v_2129S = traffic_dict.get("03F2129S", 490.0)

    debug_logs.append(
        f"[3/3] ✅ 車流計算完成 (等比換算一小時): 2100N={v_2100N:.0f}, 2100S={v_2100S:.0f}, 2125N={v_2125N:.0f}, 2129S={v_2129S:.0f}"
    )

    # (D) 安全計算差分欄位
    last_pm25 = pm25
    last_pm25_diff = 0.0
    if df_history is not None and len(df_history) > 0 and "pm25" in df_history.columns:
        last_pm25 = float(df_history["pm25"].iloc[-1])
        if "pm25_diff" in df_history.columns:
            last_pm25_diff = float(df_history["pm25_diff"].iloc[-1])

    pm25_diff = pm25 - last_pm25
    pm25_accel = pm25_diff - last_pm25_diff

    current_traffic = v_2100N + v_2100S + v_2125N + v_2129S
    last_traffic = current_traffic
    last_traffic_diff = 0.0

    gantry_cols = ["03F2100N", "03F2100S", "03F2125N", "03F2129S"]
    if df_history is not None and len(df_history) > 0 and all(col in df_history.columns for col in gantry_cols):
        last_traffic = float(df_history[gantry_cols].iloc[-1].sum())
        if "traffic_diff" in df_history.columns:
            last_traffic_diff = float(df_history["traffic_diff"].iloc[-1])

    traffic_diff = current_traffic - last_traffic
    traffic_accel = traffic_diff - last_traffic_diff

    cur_h = now.hour
    hour_sin = float(np.sin(2 * np.pi * cur_h / 24.0))
    hour_cos = float(np.cos(2 * np.pi * cur_h / 24.0))

    features = [
        press, temp, rh, wind_spd, wind_dir, rain,
        pm25, pm25_diff, pm25_accel,
        traffic_diff, traffic_accel,
        v_2100N, v_2100S, v_2125N, v_2129S,
        hour_sin, hour_cos,
    ]

    return features, debug_logs
