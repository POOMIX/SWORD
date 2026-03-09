"""
hybrid_nids.py — Multiclass Hybrid NIDS (DoS / PortScan / Benign)
ปิด Statistical และ Behavioral analysis
ใช้ ML ensemble: Decision Tree + Random Forest + XGBoost
"""

import os
import json
import time
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from dotenv import load_dotenv

from utils.adaptive_flow_features import AdaptiveFlowFeatureExtractor
from utils.session_manager import SessionManager
from utils.flow_finalizer import FlowFinalizer
from suricata.suricata_parser import SuricataParser
from telegram_module.telegram_alert import TelegramAlerter as TelegramAlert

load_dotenv()

# ─────────────────────────────────────────────
# กลุ่ม Label ที่สนใจ
# ─────────────────────────────────────────────
LABEL_MAP = {
    # BENIGN
    "BENIGN": "Benign",

    # DoS / DDoS
    "DoS Hulk":         "DoS",
    "DoS GoldenEye":    "DoS",
    "DoS slowloris":    "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS":             "DoS",

    # PortScan — CSV ใช้ชื่อนี้ (ตรวจสอบแล้วจาก check_labels.py)
    "PortScan":         "PortScan",
    "Port Scan":        "PortScan",   # กันเผื่อ variant
}

# class ที่ต้องการเท่านั้น  (ตัวที่ไม่อยู่ใน set นี้จะถูก drop)
TARGET_CLASSES = {"Benign", "DoS", "PortScan"}

# ─────────────────────────────────────────────
# Network Constants (documented, not arbitrary)
# ─────────────────────────────────────────────
# ไม่ hardcode header bytes เพราะ TCP Options ต่างกันตาม OS/kernel
# ใช้ per-packet ratio features แทน absolute bytes
# เพื่อให้ robust ต่อ OS ที่ต่างกัน (Linux/Windows/macOS)

# Minimum flow duration to avoid division-by-zero (1 µs)
MIN_FLOW_DURATION_S: float = 1e-6

# Ensemble model weights — tuned empirically on CICIDS2017 validation set
# XGBoost weighted higher due to superior F1 on minority classes
DT_WEIGHT:  float = 0.20
RF_WEIGHT:  float = 0.30
XGB_WEIGHT: float = 0.50


# ─────────────────────────────────────────────
# Feature ที่ใช้ train (ต้องตรงกับ Suricata extractor)
# ─────────────────────────────────────────────
FEATURES = [
    # Flow-level features — ทั้งหมดนี้ Suricata flow event ให้ได้จริง
    "dest_port",              # port ปลายทาง
    "duration",               # ระยะเวลา flow (microseconds)
    "total_fwd_packets",      # packets ขาไป
    "total_bwd_packets",      # packets ขากลับ
    "flow_packets_per_sec",   # packet rate
    "down_up_ratio",          # bwd/fwd bytes ratio
    # Engineered features — ratio-based ไม่พึ่ง absolute bytes
    # จึง robust ต่อ OS/kernel ที่ต่างกัน (TCP Options ต่างกัน)
    "total_packets",          # fwd+bwd packets รวม
    "fwd_bwd_ratio",          # fwd_pkts/(bwd_pkts+1)
    "duration_ms",            # duration เป็น ms
    "fwd_bytes_per_pkt",      # bytes ต่อ packet ขาไป  ← แทน absolute bytes
    "bwd_bytes_per_pkt",      # bytes ต่อ packet ขากลับ ← แทน absolute bytes
    "bytes_ratio",            # fwd_bytes_per_pkt / (bwd_bytes_per_pkt+1)
    "pkt_size_ratio",         # avg_fwd_size / avg_bwd_size
    "flow_bytes_per_pkt",     # total bytes / total packets
]

# ─────────────────────────────────────────────
# CICIDS2017 column name → internal feature name
# ─────────────────────────────────────────────
CICIDS_COL_MAP = {
    # Raw CICIDS2017 columns → internal names
    "Destination Port":            "dest_port",
    "Flow Duration":               "duration",
    "Total Fwd Packets":           "total_fwd_packets",
    "Total Backward Packets":      "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Flow Bytes/s":                "flow_bytes_per_sec",
    "Flow Packets/s":              "flow_packets_per_sec",
    "Down/Up Ratio":               "down_up_ratio",
    # Label
    "Label":                       "Label",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hybrid_nids.log"),
    ],
)
logger = logging.getLogger("hybrid-nids")


# ═══════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════

def _normalize_label_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize label column:
    - strip whitespace จาก column names ทั้งหมด
    - รองรับทั้ง 'Label', ' Label', 'label' (case-insensitive)
    """
    df.columns = df.columns.str.strip()
    # หา label column แบบ case-insensitive
    for col in df.columns:
        if col.lower() == "label":
            if col != "Label":
                df = df.rename(columns={col: "Label"})
            break
    return df


def load_and_preprocess(dataset_path: str) -> pd.DataFrame:
    """
    โหลด CICIDS2017 CSV (ไฟล์เดียวหรือ directory)
    แล้วกรองเฉพาะ Benign / DoS / PortScan
    รองรับ column name ที่ต่างกัน: 'Label', ' Label', 'label'
    """
    p = Path(dataset_path)
    if p.is_dir():
        csv_files = list(p.glob("*.csv"))
        logger.info(f"พบ {len(csv_files)} ไฟล์ CSV")
        dfs = []
        for f in csv_files:
            logger.info(f"  โหลด: {f.name}")
            tmp = pd.read_csv(f, low_memory=False)
            tmp = _normalize_label_col(tmp)
            if "Label" not in tmp.columns:
                logger.warning(f"  [SKIP] {f.name}: ไม่พบ Label column (columns: {list(tmp.columns[-5:])})")
                continue
            dfs.append(tmp)
            labels_found = tmp["Label"].str.strip().unique()
            logger.info(f"  labels: {sorted(str(x) for x in labels_found)}")
        if not dfs:
            raise ValueError("ไม่พบไฟล์ CSV ที่มี Label column เลย")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p, low_memory=False)
        df = _normalize_label_col(df)

    # เลือกเฉพาะ column ที่ต้องการ (CICIDS_COL_MAP keys ถูก strip แล้ว)
    available_cicids_cols = [c for c in CICIDS_COL_MAP if c in df.columns]
    df = df[available_cicids_cols].copy()
    df.rename(columns=CICIDS_COL_MAP, inplace=True)

    # ── Feature Engineering ──────────────────────────────────────────────────
    # ใช้ per-packet ratio features แทน absolute bytes
    # เพราะ CICIDS2017 (CICFlowMeter) วัด payload-only bytes
    # แต่ Suricata วัด full-packet bytes รวม TCP Options ที่ต่างกันตาม OS/kernel
    # ratio-based features ไม่ขึ้นกับ header overhead จึง robust กว่า

    fwd_pkts = df["total_fwd_packets"].replace(0, 1)
    bwd_pkts = df["total_bwd_packets"].replace(0, 1)
    fwd_bytes = df["total_fwd_bytes"]
    bwd_bytes = df["total_bwd_bytes"]
    total_pkts = df["total_fwd_packets"] + df["total_bwd_packets"]

    # per-packet bytes (normalize absolute bytes ด้วย packet count)
    df["fwd_bytes_per_pkt"]  = fwd_bytes / fwd_pkts
    df["bwd_bytes_per_pkt"]  = bwd_bytes / bwd_pkts

    # ratio features
    df["total_packets"]      = total_pkts
    df["fwd_bwd_ratio"]      = df["total_fwd_packets"] / (df["total_bwd_packets"] + 1)
    df["bytes_ratio"]        = df["fwd_bytes_per_pkt"] / (df["bwd_bytes_per_pkt"] + 1)
    df["pkt_size_ratio"]     = df["fwd_bytes_per_pkt"] / (df["bwd_bytes_per_pkt"] + 1)

    total_bytes = fwd_bytes + bwd_bytes
    df["flow_bytes_per_pkt"] = total_bytes / total_pkts.replace(0, 1)
    df["duration_ms"]        = df["duration"] / 1000.0

    # map label → group (strip whitespace ก่อน map)
    df["Label"] = df["Label"].astype(str).str.strip().map(LABEL_MAP)
    df.dropna(subset=["Label"], inplace=True)
    df = df[df["Label"].isin(TARGET_CLASSES)].copy()

    # clean numeric
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.drop_duplicates(inplace=True)

    logger.info(f"Dataset หลัง filter:\n{df['Label'].value_counts().to_string()}")
    return df


def balance_binary(df: pd.DataFrame, attack: str, ratio: float = 2.0) -> pd.DataFrame:
    """
    Balance dataset สำหรับ binary task (Benign vs attack)
    - ใช้ Benign มากกว่า attack ไม่เกิน ratio เท่า (default 2:1)
    - ป้องกัน overfit จาก majority class และ underfit จาก minority class
    ratio=2.0 หมายความว่า Benign:Attack = 2:1
    """
    benign  = df[df["Label"] == "Benign"]
    attacks = df[df["Label"] == attack]
    n_attack = len(attacks)
    n_benign = min(len(benign), int(n_attack * ratio))

    logger.info(f"  [{attack}] balance: Benign={n_benign:,} | {attack}={n_attack:,} (ratio {ratio:.1f}:1)")

    benign_sampled = benign.sample(n_benign, random_state=42)
    balanced = pd.concat([benign_sampled, attacks], ignore_index=True)
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def _train_one_binary(X_tr, X_te, y_tr, y_te, label: str, param_grid: dict):
    """
    Train DT + RF + XGBoost สำหรับ binary task หนึ่งตัว
    Benign=0, Attack=1
    คืน (dt, rf, xgb)
    """
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=20, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    dt.fit(X_tr, y_tr)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        random_state=42, class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    # XGBoost binary + GridSearch
    xgb_base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
    )
    grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    xgb = grid.best_estimator_
    logger.info(f"  [{label}] XGBoost best params: {grid.best_params_}")

    # Evaluate
    for name, model in [("DT", dt), ("RF", rf), ("XGB", xgb)]:
        y_pred = model.predict(X_te)
        report = classification_report(
            y_te, y_pred,
            target_names=["Benign", label],
            digits=4,
        )
        cm = confusion_matrix(y_te, y_pred)
        logger.info(f"\n{'='*50}\n[{label}] {name}\n{report}")
        logger.info(f"  Confusion Matrix:\n  TN={cm[0,0]} FP={cm[0,1]} | FN={cm[1,0]} TP={cm[1,1]}")

    return dt, rf, xgb


def train_models(dataset_path: str, model_dir: str = "./model"):
    """
    OvR Binary Training พร้อม balance strategy:
    - ชุดที่ 1: Benign vs PortScan  (Benign:PortScan = 2:1)
    - ชุดที่ 2: Benign vs DoS       (Benign:DoS = 2:1)
    ratio 2:1 ป้องกัน overfit (ไม่เท่ากันเลย) และ underfit (ไม่ห่างเกินไป)
    บันทึก model แยกใน model_dir/portscan/ และ model_dir/dos/
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess(dataset_path)
    logger.info(f"Dataset ทั้งหมด:\n{df['Label'].value_counts().to_string()}")

    # StandardScaler fit บน Benign ทั้งหมด เพื่อให้ scale สะท้อน normal traffic
    X_all = df[FEATURES].values
    scaler = StandardScaler()
    scaler.fit(X_all)
    X_all_s = scaler.transform(X_all)
    df_scaled = pd.DataFrame(X_all_s, columns=FEATURES)
    df_scaled["Label"] = df["Label"].values

    param_grid = {
        "max_depth":     [6, 9],
        "n_estimators":  [100, 150],
        "learning_rate": [0.05, 0.1],
    }

    attack_types = [("PortScan", "portscan"), ("DoS", "dos")]

    for attack_label, attack_dir in attack_types:
        logger.info(f"\n{'='*50}\nTraining Benign vs {attack_label}...")
        sub_dir = os.path.join(model_dir, attack_dir)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

        # balance Benign:Attack = 2:1
        df_binary = balance_binary(df_scaled, attack_label, ratio=2.0)
        X_sub = df_binary[FEATURES].values
        y_sub = (df_binary["Label"] == attack_label).astype(int).values

        logger.info(f"  หลัง balance: total={len(df_binary):,} | Benign={( y_sub==0).sum():,} | {attack_label}={(y_sub==1).sum():,}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
        )

        dt, rf, xgb = _train_one_binary(X_tr, X_te, y_tr, y_te, attack_label, param_grid)

        for fname, obj in [
            ("dt_model.pkl", dt), ("rf_model.pkl", rf), ("xgb_model.pkl", xgb),
        ]:
            with open(os.path.join(sub_dir, fname), "wb") as f:
                pickle.dump(obj, f)
        logger.info(f"  ✅ บันทึก {attack_label} models → {sub_dir}")

    # บันทึก shared scaler + features
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump(FEATURES, f)

    logger.info(f"\n✅ OvR Training เสร็จ | models: {model_dir}/portscan/ และ {model_dir}/dos/")
    return [a for a, _ in attack_types]


# ═══════════════════════════════════════════════════════
#  HYBRID NIDS (Inference)
# ═══════════════════════════════════════════════════════

class HybridNIDS:
    """
    Real-time NIDS: อ่าน Suricata eve.json → ทำนาย attack type
    Statistical + Behavioral ปิดไว้ก่อน
    """

    def __init__(
        self,
        model_dir:    str  = "./model",
        use_telegram: bool = False,
    ):
        self.model_dir   = model_dir
        self.use_telegram = use_telegram

        self._load_models()

        self.parser   = SuricataParser()
        self.session_mgr = SessionManager()
        self.extractor   = AdaptiveFlowFeatureExtractor(selected_features=self.features)
        self.finalizer   = FlowFinalizer(
            anomaly_detector=self,          # ส่ง self เพื่อให้ finalizer เรียก predict
            use_statistical=False,          # ปิด stat
            use_behavioral=False,           # ปิด behavioral
        )

        if use_telegram:
            token   = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            self.telegram = TelegramAlert(token, chat_id) if token and chat_id else None
        else:
            self.telegram = None

        self.alert_count = 0



        logger.info("HybridNIDS พร้อมทำงาน (OvR Binary: Suricata-first, ML-fallback)")

    # ── Model loading ──────────────────────────────────

    def _load_models(self):
        def _load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        md = self.model_dir
        self.scaler = _load(os.path.join(md, "scaler.pkl"))
        with open(os.path.join(md, "features.json")) as f:
            self.features = json.load(f)

        # OvR: โหลด model แยกสำหรับแต่ละ attack type
        self.models_ovr = {}
        for attack in ["portscan", "dos"]:
            sub = os.path.join(md, attack)
            self.models_ovr[attack] = {
                "dt":  _load(os.path.join(sub, "dt_model.pkl")),
                "rf":  _load(os.path.join(sub, "rf_model.pkl")),
                "xgb": _load(os.path.join(sub, "xgb_model.pkl")),
                "label": "PortScan" if attack == "portscan" else "DoS",
            }

        logger.info(f"โหลด OvR models สำเร็จ | PortScan + DoS binary classifiers")

    # ── ML Prediction (multiclass) ────────────────────

    def predict(self, features: pd.DataFrame) -> dict:
        """
        OvR Binary Prediction:
        - Benign vs PortScan → P(PortScan)
        - Benign vs DoS      → P(DoS)
        เลือก class ที่ combined probability สูงที่สุด ถ้าเกิน THRESHOLD=0.5
        """
        THRESHOLD = 0.5
        try:
            X = features.reindex(columns=self.features, fill_value=0).values
            X_scaled = self.scaler.transform(X)

            scores    = {}  # label → combined prob
            per_model = {}  # label → {dt, rf, xgb conf}
            votes_out = {}  # label → {dt, rf, xgb class}

            for attack, m in self.models_ovr.items():
                label  = m["label"]
                dt_p   = float(m["dt"].predict_proba(X_scaled)[0][1])
                rf_p   = float(m["rf"].predict_proba(X_scaled)[0][1])
                xgb_p  = float(m["xgb"].predict_proba(X_scaled)[0][1])

                # weighted average DT=20% RF=30% XGB=50%
                combined = DT_WEIGHT * dt_p + RF_WEIGHT * rf_p + XGB_WEIGHT * xgb_p
                scores[label]    = combined
                per_model[label] = {"dt": dt_p, "rf": rf_p, "xgb": xgb_p}
                votes_out[label] = {
                    "dt":  label if dt_p  >= 0.5 else "Benign",
                    "rf":  label if rf_p  >= 0.5 else "Benign",
                    "xgb": label if xgb_p >= 0.5 else "Benign",
                }

            # เลือก attack ที่ score สูงสุด
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]

            if best_score >= THRESHOLD:
                pred_class  = best_label
                is_attack   = True
                confidence  = best_score
                model_votes = votes_out[best_label]
                model_conf  = per_model[best_label]
            else:
                pred_class  = "Benign"
                is_attack   = False
                confidence  = 1.0 - best_score
                model_votes = {"dt": "Benign", "rf": "Benign", "xgb": "Benign"}
                model_conf  = per_model[best_label]

            return {
                "predicted_class": pred_class,
                "confidence":      confidence,
                "is_attack":       is_attack,
                "proba_per_class": {**scores, "Benign": 1.0 - best_score},
                "model_votes":     model_votes,
                "model_conf":      model_conf,
            }

        except Exception as e:
            logger.error(f"predict() error: {e}")
            return {
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "is_attack": False,
                "proba_per_class": {},
                "model_votes": {},
                "model_conf": {},
            }

    # ── Alert handler ──────────────────────────────────

    def handle_alert(self, result: dict):
        """
        จัดการ alert เมื่อตรวจพบการโจมตี
        """
        self.alert_count += 1
        ts      = datetime.fromtimestamp(result.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S")
        src     = result.get("src_ip", "?")
        dst     = result.get("dst_ip", "?")
        dport   = result.get("dst_port", "?")
        proto   = result.get("proto", "?")
        cls     = result.get("predicted_class", "Unknown")
        conf    = result.get("confidence", 0.0)
        votes   = result.get("model_votes", {})

        method = result.get("detection_method", "ml")
        if method == "suricata_signature":
            sig = result.get("suricata_signature", "")
            cat = result.get("suricata_category", "")
            sev = result.get("suricata_severity", "")
            msg = (
                f"🚨 [{self.alert_count}] {ts} | Suricata Alert | "
                f"{src} → {dst}:{dport}/{proto} | "
                f"severity={sev} | category={cat} | "
                f"signature={sig}"
            )
        else:
            msg = (
                f"🚨 [{self.alert_count}] {ts} | {cls} | "
                f"{src} → {dst}:{dport}/{proto} | "
                f"confidence={conf:.2%} | votes={votes}"
            )
        logger.warning(msg)

        # บันทึกลง CSV
        self._append_csv(result)

        # Telegram
        if self.telegram:
            try:
                self.telegram.send_alert(msg)
            except Exception as e:
                logger.error(f"Telegram error: {e}")

    def _append_csv(self, result: dict, path: str = "flow_results.csv"):
        row = {
            "timestamp":          result.get("timestamp"),
            "src_ip":             result.get("src_ip"),
            "dst_ip":             result.get("dst_ip"),
            "dst_port":           result.get("dst_port"),
            "proto":              result.get("proto"),
            "predicted_class":    result.get("predicted_class"),
            "confidence":         result.get("confidence"),
            "detection_method":   result.get("detection_method", "ml"),
            "suricata_signature": result.get("suricata_signature", ""),
            "dt_vote":            result.get("model_votes", {}).get("dt"),
            "rf_vote":            result.get("model_votes", {}).get("rf"),
            "xgb_vote":           result.get("model_votes", {}).get("xgb"),
            "duration":           result.get("duration"),
            "total_bytes":        result.get("total_bytes"),
            "total_packets":      result.get("total_packets"),
        }
        header = not Path(path).exists()
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

    # ── Real-time monitoring ───────────────────────────

    def monitor_realtime(self, eve_json_path: str):
        """
        ติดตาม Suricata eve.json แบบ real-time (tail -f style)
        กด Ctrl+C เพื่อหยุด
        """
        logger.info(f"เริ่ม monitor: {eve_json_path}")
        logger.info("กด Ctrl+C เพื่อหยุด")
        try:
            with open(eve_json_path, "r") as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    self._process_line(line.strip())
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  หยุด monitor | alerts ทั้งหมด: {self.alert_count} | ผลที่ flow_results.csv")

    def _process_line(self, line: str):
        if not line:
            return
        try:
            # ── ตรวจ Suricata alert event ก่อน ML ─────────────────────────
            # ถ้า Suricata มี signature match → alert ทันที ข้าม ML
            raw = json.loads(line)
            if raw.get("event_type") == "alert":
                self._handle_suricata_alert(raw)
                return
        except (json.JSONDecodeError, Exception):
            pass

        try:
            event = self.parser.process_line(line)
            if event is None:
                return

            # SuricataFlow มี type_ = 'conn' และ flow_source = 'suricata'
            is_flow = (
                hasattr(event, "type_") and getattr(event, "type_", "") == "conn"
            ) or (
                hasattr(event, "flow_source")
            )

            if is_flow:
                # กรอง traffic ปกติที่ไม่ใช่ attack
                src = getattr(event, "saddr", "") or ""
                dst = getattr(event, "daddr", "") or ""

                skip = (
                    src.startswith("fe80:") or src.startswith("ff") or
                    dst.startswith("ff02:") or dst.startswith("ff0") or
                    dst == "255.255.255.255" or
                    dst in ("224.0.0.251", "239.255.255.250", "224.0.0.252") or
                    src in ("0.0.0.0", "0000:0000:0000:0000:0000:0000:0000:0000")
                )
                if not skip:
                    self._analyze_event_direct(event)
                return

            finished_sessions = self.session_mgr.process_event(event)
            for session in finished_sessions:
                self._analyze_session(session)
        except Exception as e:
            logger.debug(f"process_line error: {e}")

    def _handle_suricata_alert(self, raw: dict):
        """
        Forward Suricata signature alert โดยตรง — ไม่ผ่าน ML ไม่ classify attack type
        """
        try:
            alert_info = raw.get("alert", {})
            signature  = alert_info.get("signature", "Unknown")
            category   = alert_info.get("category", "")
            severity   = alert_info.get("severity", 3)

            src_ip   = raw.get("src_ip", "")
            dst_ip   = raw.get("dest_ip", "")
            dst_port = raw.get("dest_port", "")
            proto    = raw.get("proto", "")

            result = {
                "timestamp":          time.time(),
                "src_ip":             src_ip,
                "dst_ip":             dst_ip,
                "dst_port":           dst_port,
                "proto":              proto,
                "predicted_class":    "Suricata Alert",
                "confidence":         1.0,
                "is_attack":          True,
                "detection_method":   "suricata_signature",
                "suricata_signature": signature,
                "suricata_category":  category,
                "suricata_severity":  severity,
                "model_votes":        {"suricata": signature},
                "total_bytes":        0,
                "total_packets":      0,
                "duration":           0,
            }
            self.handle_alert(result)
        except Exception as e:
            logger.debug(f"_handle_suricata_alert error: {e}")

    def _analyze_event_direct(self, event):
        """
        วิเคราะห์ SuricataFlow โดยตรง
        SuricataFlow fields: saddr, sport, daddr, dport, proto,
                             spkts, dpkts, sbytes, dbytes, dur, state
        """
        try:
            import pandas as pd

            # ดึง field ตรงจาก SuricataFlow dataclass
            fwd_pkts  = float(getattr(event, "spkts",  0) or 0)
            bwd_pkts  = float(getattr(event, "dpkts",  0) or 0)
            # ใช้ Suricata bytes ตรงๆ — training ก็บวก header เข้าไปแล้วเช่นกัน
            fwd_bytes = float(getattr(event, "sbytes", 0) or 0)
            bwd_bytes = float(getattr(event, "dbytes", 0) or 0)
            dur_raw   = float(getattr(event, "dur",    0) or 0)

            # คำนวณ duration จาก starttime/endtime ถ้า dur_raw = 0
            if dur_raw == 0:
                try:
                    from dateutil import parser as dp
                    t0 = dp.parse(getattr(event, "starttime", ""))
                    t1 = dp.parse(getattr(event, "endtime",   ""))
                    dur_raw = max((t1 - t0).total_seconds(), MIN_FLOW_DURATION_S)
                except Exception:
                    dur_raw = MIN_FLOW_DURATION_S

            # แปลงหน่วย
            dur_us       = dur_raw * 1e6
            dur_ms       = dur_raw * 1000.0
            dur_for_rate = max(dur_raw, MIN_FLOW_DURATION_S)

            total_bytes = fwd_bytes + bwd_bytes
            total_pkts  = fwd_pkts  + bwd_pkts
            safe_pkts   = max(total_pkts, 1)

            safe_fwd = max(fwd_pkts, 1)
            safe_bwd = max(bwd_pkts, 1)
            fwd_bpp  = fwd_bytes / safe_fwd
            bwd_bpp  = bwd_bytes / safe_bwd

            row = {
                "dest_port":            int(getattr(event, "dport", 0) or 0),
                "duration":             dur_us,
                "total_fwd_packets":    fwd_pkts,
                "total_bwd_packets":    bwd_pkts,
                "flow_packets_per_sec": total_pkts  / dur_for_rate,
                "down_up_ratio":        bwd_bytes   / max(fwd_bytes, 1),
                "total_packets":        total_pkts,
                "fwd_bwd_ratio":        fwd_pkts    / max(bwd_pkts + 1, 1),
                "duration_ms":          dur_ms,
                "fwd_bytes_per_pkt":    fwd_bpp,
                "bwd_bytes_per_pkt":    bwd_bpp,
                "bytes_ratio":          fwd_bpp / (bwd_bpp + 1),
                "pkt_size_ratio":       fwd_bpp / (bwd_bpp + 1),
                "flow_bytes_per_pkt":   total_bytes / safe_pkts,
            }

            features_df = pd.DataFrame([row])
            result = self.predict(features_df)
            result.update({
                "timestamp":        time.time(),
                "src_ip":           getattr(event, "saddr", ""),
                "dst_ip":           getattr(event, "daddr", ""),
                "dst_port":         getattr(event, "dport", ""),
                "proto":            getattr(event, "proto", ""),
                "duration":         dur_us,
                "total_bytes":      int(total_bytes),
                "total_packets":    int(total_pkts),
                "detection_method": "ml",
            })

            logger.debug(
                f"flow: {getattr(event,'saddr','')}→{getattr(event,'daddr','')}:{getattr(event,'dport','')} "
                f"| {result['predicted_class']} ({result['confidence']:.1%})"
            )

            if result["is_attack"]:
                self.handle_alert(result)

        except Exception as e:
            logger.debug(f"_analyze_event_direct error: {e}")

    def _analyze_session(self, session):
        try:
            features_df = self.extractor.extract_from_flow(session)
            if features_df is None or features_df.empty:
                return

            result = self.predict(features_df)

            # เตรียม result dict สมบูรณ์
            result.update({
                "timestamp":     getattr(session, "starttime", time.time()),
                "src_ip":        getattr(session, "saddr", ""),
                "dst_ip":        getattr(session, "daddr", ""),
                "dst_port":      getattr(session, "dport", ""),
                "proto":         getattr(session, "proto", ""),
                "duration":      getattr(session, "duration", 0),
                "total_bytes":   (getattr(session, "total_fwd_bytes", 0) +
                                  getattr(session, "total_bwd_bytes", 0)),
                "total_packets": (getattr(session, "total_fwd_packets", 0) +
                                  getattr(session, "total_bwd_packets", 0)),
            })

            if result["is_attack"]:
                self.handle_alert(result)

        except Exception as e:
            logger.debug(f"_analyze_session error: {e}")

    # ── Analyze existing log file ──────────────────────

    def analyze_file(self, eve_json_path: str, output_path: str = None):
        logger.info(f"วิเคราะห์ไฟล์: {eve_json_path}")
        with open(eve_json_path, "r") as f:
            for line in f:
                self._process_line(line.strip())

        # flush sessions ที่ยังค้างอยู่
        # ลอง flush_all() → get_all_sessions() → ตาม method ที่มีใน SessionManager
        remaining = []
        for method_name in ("flush_all", "get_all_sessions", "finalize_all", "get_active_sessions"):
            method = getattr(self.session_mgr, method_name, None)
            if method is not None:
                try:
                    remaining = method() or []
                    logger.info(f"flush ด้วย {method_name}() → {len(remaining)} sessions")
                except Exception as e:
                    logger.debug(f"{method_name}() failed: {e}")
                break
        else:
            logger.debug("SessionManager ไม่มี flush method — ข้ามขั้นตอนนี้")

        for session in remaining:
            self._analyze_session(session)

        logger.info(f"✅ วิเคราะห์เสร็จ | alerts: {self.alert_count} | ผลลัพธ์ที่ flow_results.csv")

        if output_path:
            import shutil
            shutil.copy("flow_results.csv", output_path)


# ═══════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid NIDS — Multiclass (DoS / PortScan / Benign)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",    metavar="DATASET_PATH",
                       help="Train models จาก CICIDS2017 directory หรือ CSV")
    group.add_argument("--realtime", metavar="EVE_JSON",
                       help="Monitor Suricata eve.json แบบ real-time")
    group.add_argument("--analyze",  metavar="EVE_JSON",
                       help="วิเคราะห์ไฟล์ eve.json ที่มีอยู่")

    parser.add_argument("--model_dir", default="./model",
                        help="Directory เก็บ model (default: ./model)")
    parser.add_argument("--output",    default=None,
                        help="Path สำหรับบันทึกผลลัพธ์ (--analyze mode)")
    parser.add_argument("--telegram",  action="store_true",
                        help="เปิดใช้ Telegram notifications")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        classes = train_models(args.train, args.model_dir)
        logger.info(f"Training เสร็จ | classes: {classes}")
        return

    nids = HybridNIDS(model_dir=args.model_dir, use_telegram=args.telegram)

    if args.realtime:
        nids.monitor_realtime(args.realtime)
    elif args.analyze:
        nids.analyze_file(args.analyze, args.output)


if __name__ == "__main__":
    main()
