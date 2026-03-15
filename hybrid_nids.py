"""
hybrid_nids.py — OvR Binary Hybrid NIDS (DoS / PortScan / Benign)
v3.2 — Isolated Training & Per-task Scaling
ใช้ ML ensemble: Decision Tree + Random Forest + XGBoost

Changelog v3.2:
- [NEW] Isolated Task: แยกไฟล์เทรนตามประเภทการโจมตี (Benign ไม่ปนกัน)
- [NEW] Per-task Scaler: บันทึกและโหลด Scaler แยกตาม attack type
- [FIX] Inference: predict() เลือกใช้ Scaler ที่ตรงกับ task นั้นๆ
"""

VERSION = "3.2-isolated-features"

import os
import json
import time
import signal
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# กลุ่ม Label ที่สนใจ
# ─────────────────────────────────────────────
LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "PortScan",
    "Port Scan": "PortScan",
}

TARGET_CLASSES = {"Benign", "DoS", "PortScan"}

# ─────────────────────────────────────────────
# ML Constants
# ─────────────────────────────────────────────
MIN_FLOW_DURATION_S: float = 1e-6
DT_WEIGHT:  float = 0.20
RF_WEIGHT:  float = 0.30
XGB_WEIGHT: float = 0.50
DEFAULT_THRESHOLD: float = 0.5

# ─────────────────────────────────────────────
# Feature Selection
# ─────────────────────────────────────────────
FEATURES = [
    # Original Features
    "dest_port", 
    "duration", 
    "total_fwd_packets", 
    "total_bwd_packets", 
    "total_packets", 
    "flow_packets_per_sec",
    # Derived Features
    "duration_ms", 
    "fwd_bwd_ratio", 
    "pkt_ratio", 
    "has_response", 
    "flow_iat_mean",
    "is_long_connection", 
    "log_duration", 
    "pkts_per_duration",
    "acc_age", 
    "n_flushes", 
    "log_acc_age"
]

CICIDS_COL_MAP = {
    "Destination Port": "dest_port",
    "Flow Duration": "duration",
    "Total Backward Packets": "total_bwd_packets",
    "Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    "Label": "Label",
}

ML_LOG_DIR = "/var/log/sword_detection"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_nids.log")],
)
logger = logging.getLogger("hybrid-nids")

# ═══════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════

def _normalize_label_col(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col.lower() == "label":
            if col != "Label": df = df.rename(columns={col: "Label"})
            break
    return df

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    MAP = {
        "Destination Port": "dest_port",
        "Flow Duration": "duration",
        "Total Fwd Packets": "total_fwd_packets",
        "Total Backward Packets": "total_bwd_packets",
        "Flow Packets/s": "flow_packets_per_sec",
        "Flow IAT Mean": "flow_iat_mean"
    }
    df.rename(columns=MAP, inplace=True)

    label_found = False
    for col in df.columns:
        if col.lower() == "label":
            df.rename(columns={col: "Label"}, inplace=True)
            label_found = True
            break
    
    if not label_found:
        raise KeyError(f" หาคอลัมน์ 'Label' ไม่เจอในไฟล์ {os.path.basename(file_path)} | คอลัมน์ที่มี: {list(df.columns)}")

    required = ["total_fwd_packets", "total_bwd_packets", "duration"]
    for r in required:
        if r not in df.columns:
            raise KeyError(f" ไม่พบคอลัมน์ '{r}' | ลองเช็คชื่อใน CSV ดูครับว่าตรงกับ {MAP} หรือไม่")

    total_pkts = df["total_fwd_packets"] + df["total_bwd_packets"]
    df["total_packets"] = total_pkts
    df["duration_ms"] = df["duration"] / 1000.0
    df["flow_packets_per_sec"] = total_pkts / (df["duration"].replace(0, 1) / 1e6)
    df["fwd_bwd_ratio"] = df["total_fwd_packets"] / (df["total_bwd_packets"] + 1)
    df["pkt_ratio"] = df["total_bwd_packets"] / (df["total_fwd_packets"].replace(0, 1))
    df["has_response"] = (df["total_bwd_packets"] > 0).astype(float)
    
    if "flow_iat_mean" not in df.columns: 
        df["flow_iat_mean"] = 0.0

    df["is_long_connection"] = (df["duration"] > 1_000_000).astype(float)
    df["log_duration"] = np.log10(df["duration"].clip(lower=1))
    df["pkts_per_duration"] = df["total_packets"] / df["log_duration"].replace(0, 1)

    duration_s = df["duration"] / 1e6
    df["acc_age"] = duration_s
    df["n_flushes"] = np.ceil(duration_s / 30.0).clip(lower=1)
    df["log_acc_age"] = np.log10(duration_s.clip(lower=1e-6) + 1)

    df["Label"] = df["Label"].astype(str).str.strip().map(LABEL_MAP)
    df.dropna(subset=["Label"], inplace=True)
    df = df[df["Label"].isin(TARGET_CLASSES)].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.drop_duplicates(inplace=True)
    return df

def balance_binary_arrays(X_train, y_train, ratio=2.0, random_state=42):
    rng = np.random.default_rng(random_state)
    idx_benign, idx_attack = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
    n_benign = min(len(idx_benign), int(len(idx_attack) * ratio))
    chosen_benign = rng.choice(idx_benign, size=n_benign, replace=False)
    idx_bal = np.concatenate([chosen_benign, idx_attack])
    rng.shuffle(idx_bal)
    return X_train[idx_bal], y_train[idx_bal]

def _train_one_binary(X_tr, X_te, y_tr, y_te, label, param_grid):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score

    print(f"[{label}] Training set: {len(X_tr):,} samples  "
                f"(Attack={int(y_tr.sum()):,} / Benign={int((y_tr==0).sum()):,})")

    # ── Decision Tree — จำกัด depth และ min_samples ──────────
    print(f"[{label}] Training Decision Tree...")
    dt_base = DecisionTreeClassifier(
        max_depth=5,           
        min_samples_leaf=30,    
        min_samples_split=60,   
        random_state=42,
        class_weight="balanced"
    )
    dt = CalibratedClassifierCV(dt_base, method="sigmoid", cv=5).fit(X_tr, y_tr)
    dt_pred = dt.predict(X_te)
    dt_cv = cross_val_score(dt_base, X_tr, y_tr, cv=5, scoring="f1", n_jobs=-1)
    print(f"   [{label}] DT | CV F1={dt_cv.mean():.4f} ±{dt_cv.std():.4f} | Test F1 below")
    print(f"\n{classification_report(y_te, dt_pred, target_names=['Benign', label], zero_division=0, digits=4)}")

    # ── Random Forest ─────────────────────────────────────────
    print(f"[{label}] Training Random Forest...")
    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,           
        min_samples_leaf=10,    
        max_features="sqrt",    
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5).fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_cv = cross_val_score(rf_base, X_tr, y_tr, cv=5, scoring="f1", n_jobs=-1)
    print(f"[{label}] RF | CV F1={rf_cv.mean():.4f} ±{rf_cv.std():.4f} | Test F1 below")
    print(f"\n{classification_report(y_te, rf_pred, target_names=['Benign', label], zero_division=0, digits=4)}")

    # ── XGBoost — เพิ่ม regularization ───────────────────────
    print(f"[{label}] Training XGBoost (GridSearch cv=5)...")
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        subsample=0.8,          
        colsample_bytree=0.8,   
        reg_alpha=0.1,          
        reg_lambda=1.5,         
        min_child_weight=5,     
    )
    grid = GridSearchCV(
        xgb_base, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    ).fit(X_tr, y_tr)
    xgb = grid.best_estimator_
    xgb_pred = xgb.predict(X_te)
    print(f"   [{label}] XGB | best_params={grid.best_params_} | "
                f"CV F1={grid.best_score_:.4f} | Test F1 below")
    print(f"\n{classification_report(y_te, xgb_pred, target_names=['Benign', label], zero_division=0, digits=4)}")

    # ── CV vs Test gap check ──────────────────────────────────
    from sklearn.metrics import f1_score
    test_f1_dt  = f1_score(y_te, dt_pred,  zero_division=0)
    test_f1_rf  = f1_score(y_te, rf_pred,  zero_division=0)
    test_f1_xgb = f1_score(y_te, xgb_pred, zero_division=0)

    print(f"[{label}] CV vs Test gap:")
    print(f"DT  → CV={dt_cv.mean():.4f}  Test={test_f1_dt:.4f}  gap={abs(dt_cv.mean()-test_f1_dt):.4f}"
                + ("  overfit?" if abs(dt_cv.mean()-test_f1_dt) > 0.01 else " ✅"))
    print(f"     RF  → CV={rf_cv.mean():.4f}  Test={test_f1_rf:.4f}  gap={abs(rf_cv.mean()-test_f1_rf):.4f}"
                + ("  overfit?" if abs(rf_cv.mean()-test_f1_rf) > 0.01 else " ✅"))
    print(f"     XGB → CV={grid.best_score_:.4f}  Test={test_f1_xgb:.4f}  gap={abs(grid.best_score_-test_f1_xgb):.4f}"
                + ("  overfit?" if abs(grid.best_score_-test_f1_xgb) > 0.01 else " ✅"))

    return dt, rf, xgb

def _auto_tune_threshold(dt, rf, xgb, X_val, y_val, label):
    from sklearn.metrics import f1_score
    dt_p  = dt.predict_proba(X_val)[:, 1]
    rf_p  = rf.predict_proba(X_val)[:, 1]
    xgb_p = xgb.predict_proba(X_val)[:, 1]
    ensemble_p = DT_WEIGHT * dt_p + RF_WEIGHT * rf_p + XGB_WEIGHT * xgb_p

    best_t, best_f1 = 0.5, 0.0
    rows = []
    for t in [round(x * 0.05, 2) for x in range(6, 19)]:
        f1 = f1_score(y_val, (ensemble_p >= t).astype(int), zero_division=0)
        rows.append((t, f1))
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    thr_log = "  ".join([f"t={t:.2f}→F1={f:.3f}{'★' if t == best_t else ''}" for t, f in rows])
    print(f"[{label}] Threshold search:\n     {thr_log}")
    print(f"[{label}] Best threshold = {best_t:.2f}  (Ensemble F1 = {best_f1:.4f})")
    return best_t

def train_models(dataset_path: str, model_dir: str = "./model"):
    signal.signal(signal.SIGINT, lambda s, f: (
        print("\nTraining interrupted by user (Ctrl+C)"),
        exit(0)
    ))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ATTACK_FILE_MAP = {
        "PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "DoS": "Wednesday-workingHours.pcap_ISCX.csv"
    }

    param_grid = {
        "max_depth":     [4, 6, 9],
        "n_estimators":  [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [3, 5],    
        "subsample":     [0.7, 0.8],   
    }
    attack_types = [("PortScan", "portscan"), ("DoS", "dos")]
    tuned_thresholds = {}

    for attack_label, attack_dir in attack_types:
        target_file = ATTACK_FILE_MAP.get(attack_label)
        full_path = os.path.join(dataset_path, target_file)
        if not os.path.exists(full_path):
            print(f"Not found {target_file} — skip {attack_label}")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {attack_label}  ←  {target_file}")
        print(f"{'='*60}")

        t0 = time.time()
        df_binary = load_and_preprocess(full_path)
        df_binary = df_binary[df_binary["Label"].isin(["Benign", attack_label])].copy()

        vc = df_binary["Label"].value_counts()
        print(f"Dataset after filter: {len(df_binary):,} rows")
        for cls, cnt in vc.items():
            print(f"     {cls:<12}: {cnt:,} ({cnt/len(df_binary)*100:.1f}%)")

        X_raw = df_binary[FEATURES].values
        y_raw = (df_binary["Label"] == attack_label).astype(int).values
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

        print(f"Split → Train: {len(X_tr_raw):,}  Test: {len(X_te_raw):,}")

        task_scaler = StandardScaler()
        X_tr_scaled = task_scaler.fit_transform(X_tr_raw)
        X_te_scaled  = task_scaler.transform(X_te_raw)

        X_tr_bal, y_tr_bal = balance_binary_arrays(X_tr_scaled, y_tr)
        print(f"After balance → {len(X_tr_bal):,} "
                    f"(Attack={int(y_tr_bal.sum()):,} / Benign={int((y_tr_bal==0).sum()):,})")

        dt, rf, xgb = _train_one_binary(X_tr_bal, X_te_scaled, y_tr_bal, y_te, attack_label, param_grid)

        sub_dir = os.path.join(model_dir, attack_dir)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)
        for fn, obj in [("dt_model.pkl", dt), ("rf_model.pkl", rf),
                        ("xgb_model.pkl", xgb), ("scaler.pkl", task_scaler)]:
            with open(os.path.join(sub_dir, fn), "wb") as f:
                pickle.dump(obj, f)

        tuned_thresholds[attack_label] = _auto_tune_threshold(
            dt, rf, xgb, X_te_scaled, y_te, attack_label)

        print(f"Models saved → {sub_dir}")
        print(f"{attack_label} total time: {time.time()-t0:.1f}s")

    with open(os.path.join(model_dir, "features.json"), "w") as f: json.dump(FEATURES, f)
    with open(os.path.join(model_dir, "thresholds.json"), "w") as f: json.dump(tuned_thresholds, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete | Thresholds: {tuned_thresholds}")
    print(f"{'='*60}\n")
    return [a for a, _ in attack_types]

class HybridNIDS:
    def __init__(self, model_dir="./model", use_telegram=False, threshold=DEFAULT_THRESHOLD):
        self.model_dir, self.use_telegram, self.threshold = model_dir, use_telegram, threshold
        self._load_models()
        self.alert_count, self._alert_cache = 0, {}
        self.local_ips = self._get_local_ips()

    def _get_local_ips(self) -> set:
        import socket
        ips = {"127.0.0.1", "::1"}
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                ips.add(info[4][0].split("%")[0])
        except Exception:
            pass
        logger.info(f"Local IPs (will be ignored): {ips}")
        return ips

    def _load_models(self):
        md = self.model_dir
        self.features = json.load(open(os.path.join(md, "features.json")))
        self.models_ovr = {}
        for att in ["portscan", "dos"]:
            sd = os.path.join(md, att)
            if os.path.exists(sd):
                self.models_ovr[att] = {
                    "dt": pickle.load(open(os.path.join(sd, "dt_model.pkl"), "rb")),
                    "rf": pickle.load(open(os.path.join(sd, "rf_model.pkl"), "rb")),
                    "xgb": pickle.load(open(os.path.join(sd, "xgb_model.pkl"), "rb")),
                    "scaler": pickle.load(open(os.path.join(sd, "scaler.pkl"), "rb")),
                    "label": "PortScan" if att == "portscan" else "DoS"
                }
        self.tuned_thresholds = json.load(open(os.path.join(md, "thresholds.json"))) if os.path.exists(os.path.join(md, "thresholds.json")) else {}

    def predict(self, features: pd.DataFrame) -> dict:
        try:
            X_df = features.reindex(columns=self.features, fill_value=0)
            scores, per_model = {}, {}
            for att, m in self.models_ovr.items():
                label = m["label"]
                X_scaled = m["scaler"].transform(X_df.values)
                dt_p, rf_p, xgb_p = m["dt"].predict_proba(X_scaled)[0][1], m["rf"].predict_proba(X_scaled)[0][1], m["xgb"].predict_proba(X_scaled)[0][1]
                scores[label] = (DT_WEIGHT * dt_p) + (RF_WEIGHT * rf_p) + (XGB_WEIGHT * xgb_p)
                per_model[label] = {"dt": dt_p, "rf": rf_p, "xgb": xgb_p}

            max_s = max(scores.values()); best_l = max(scores, key=scores.get)
            eff_t = self.tuned_thresholds.get(best_l, self.threshold)

            is_atk = max_s >= eff_t
            return {
                "predicted_class": best_l if is_atk else "Benign",
                "confidence": max_s if is_atk else 1.0 - max_s,
                "is_attack": is_atk, "model_conf": per_model[best_l], "all_scores": scores, "all_per_model": per_model
            }
        except Exception as e: return {"predicted_class": "Unknown", "is_attack": False}

    def _handle_suricata_alert(self, raw):
        alert = raw.get("alert", {})
        if alert.get("category") in {"Generic Protocol Command Decode", "Not Suspicious Traffic"}: return
        res = {"timestamp": time.time(), "src_ip": raw.get("src_ip"), "dst_ip": raw.get("dest_ip"), "dst_port": raw.get("dest_port"), "proto": raw.get("proto"), "predicted_class": "Suricata Alert", "is_attack": True, "detection_method": "suricata_signature"}
        self.handle_alert(res)

    def handle_alert(self, res):
        now = time.time()
        key = (res.get("src_ip"), res.get("dst_ip"), res.get("dst_port"), res.get("predicted_class"))
        if now - self._alert_cache.get(key, 0) < 10: return
        self._alert_cache[key] = now
        self.alert_count += 1

        predicted = res.get("predicted_class", "Unknown")
        confidence = res.get("confidence", 0.0)
        all_per_model = res.get("all_per_model", {})
        all_scores = res.get("all_scores", {})

        # ── Header ──────────────────────────────────────────────
        print(
            f"🚨 [{self.alert_count}] {res.get('src_ip')} -> "
            f"{res.get('dst_ip')}:{res.get('dst_port')} | "
            f"Class: {predicted}  Conf: {confidence:.3f}"
        )

        # ── Per-task model breakdown ─────────────────────────────
        for task_label, pm in all_per_model.items():
            ensemble = all_scores.get(task_label, 0.0)
            icon = "🔴" if task_label == "DoS" else "🟡"
            print(
                f"   {icon} {task_label:<10} | "
                f"DT={pm.get('dt', 0):.3f}  "
                f"RF={pm.get('rf', 0):.3f}  "
                f"XGB={pm.get('xgb', 0):.3f}  "
                f"→ Ensemble={ensemble:.3f}"
            )

        # write to log file
        with open(Path(ML_LOG_DIR, "ml_log.json"), "a") as f:
            log_json = json.dumps({
                "src_ip": res.get("src_ip"),
                "dest_ip": res.get("dst_ip"),
                "dest_port": res.get("dst_port"),
                "confidence": f"{confidence:.3f}",
                "predicted_attack": predicted
            })
            f.write(log_json + "\n")

    def monitor_realtime(self, eve_path):
        print(f"📡 Monitoring: {eve_path}  (Ctrl+C to stop)")
        self._running = True

        def _stop(signum, frame):
            self._running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        try:
            with open(eve_path, "r") as f:
                f.seek(0, 2)
                while self._running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    self._process_line(line.strip())
        finally:
            print(f"🛑 Stopped. Total alerts: {self.alert_count}")
            

    def _process_line(self, line):
        try:
            raw = json.loads(line)
            event_type = raw.get("event_type")

            if event_type == "alert":
                self._handle_suricata_alert(raw)
                return

            if event_type == "flow":
                src_ip = raw.get("src_ip", "")
                # ── กรอง traffic ของเครื่องตัวเอง ──
                if src_ip in self.local_ips:
                    return
                self._analyze_flow(raw)
        except:
            pass

    def _analyze_flow(self, raw):
        flow     = raw.get("flow", {})
        dur_s    = float(flow.get("age", 0))
        spkts    = int(flow.get("pkts_toserver", 0))
        dpkts    = int(flow.get("pkts_toclient", 0))
        dst_port = int(raw.get("dest_port", 0))
        total    = spkts + dpkts

        dur_us = dur_s * 1e6
        log_dur = np.log10(max(dur_us, 1))

        row = {
            "dest_port":          dst_port,
            "duration":           dur_us,
            "duration_ms":        dur_s * 1000,
            "total_fwd_packets":  spkts,
            "total_bwd_packets":  dpkts,
            "total_packets":      total,
            "flow_packets_per_sec": total / max(dur_s, 1e-6),
            "fwd_bwd_ratio":      spkts / (dpkts + 1),
            "pkt_ratio":          dpkts / max(spkts, 1),
            "has_response":       float(dpkts > 0),
            "flow_iat_mean":      dur_us / max(total - 1, 1),
            "is_long_connection": float(dur_s > 1),
            "log_duration":       log_dur,
            "pkts_per_duration":  total / max(log_dur, 1),
            "acc_age":            dur_s,
            "n_flushes":          np.ceil(dur_s / 30),
            "log_acc_age":        np.log10(dur_s + 1),
        }

        res = self.predict(pd.DataFrame([row]))
        if res["is_attack"]:
            res.update({
                "src_ip":   raw.get("src_ip", ""),
                "dst_ip":   raw.get("dest_ip", ""),
                "dst_port": dst_port,
            })
            self.handle_alert(res)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", metavar="PATH")
    group.add_argument("--realtime", metavar="PATH")
    parser.add_argument("--model_dir", default="./model")
    args = parser.parse_args()

    if args.train: train_models(args.train, args.model_dir)
    elif args.realtime: 
        # Create Log dir
        Path("/var/log/sword_detection").mkdir(parents=True, exist_ok=True)
        HybridNIDS(model_dir=args.model_dir).monitor_realtime(args.realtime)

if __name__ == "__main__": main()