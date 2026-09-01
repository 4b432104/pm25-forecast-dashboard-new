import sqlite3

DB_PATH = "prediction.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 建立預測資料表 (新增 step 欄位，欄位名統一為 predicted_pm25)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        base_time TEXT,
        target_time TEXT,
        predicted_pm25 REAL,
        step INTEGER,
        UNIQUE(base_time, target_time)
    )
    """)

    # 2. 建立實測資料表 (17 特徵 + 1 timestamp = 18 欄)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS real_data (
        timestamp TEXT PRIMARY KEY,
        press REAL, temp REAL, rh REAL, wind_spd REAL, wind_dir REAL, rain REAL,
        pm25 REAL, pm25_diff REAL, pm25_accel REAL, traffic_diff REAL, traffic_accel REAL,
        v_2100N REAL, v_2100S REAL, v_2125N REAL, v_2129S REAL,
        hour_sin REAL, hour_cos REAL
    )
    """)
    conn.commit()
    conn.close()


def save_real_data(timestamp, features):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = """
    INSERT OR REPLACE INTO real_data 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(query, [timestamp] + list(features))
    conn.commit()
    conn.close()


def save_predictions(base_time, predictions):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for item in predictions:
        if len(item) == 3:
            target_time, pred_val, step = item
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions (base_time, target_time, predicted_pm25, step) 
                VALUES (?, ?, ?, ?)
                """,
                (base_time, target_time, pred_val, step)
            )
        else:
            target_time, pred_val = item
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions (base_time, target_time, predicted_pm25) 
                VALUES (?, ?, ?)
                """,
                (base_time, target_time, pred_val)
            )
            
    conn.commit()
    conn.close()