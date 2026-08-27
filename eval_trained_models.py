"""
eval_trained_models.py — เอาผล classification_report แบบเต็ม (precision/
recall/F1 ต่อ task) กลับมาดูอีกครั้ง โดย "ไม่ต้องเทรนใหม่"

เหตุผลที่ทำได้: hybrid_nids.py ใช้ random_state=42 คงที่ทุกจุด
(train_test_split ตอน train/val/test split) ดังนั้นถ้าโหลด dataset
เดียวกันด้วยโค้ด preprocessing เดียวกัน (import load_and_preprocess จาก
hybrid_nids.py ตรง ๆ ไม่ก็อปมาเขียนใหม่) จะได้ split ชุดเดิมเป๊ะ แล้ว
เอาโมเดลที่ pickle เก็บไว้แล้วมา predict ซ้ำ + threshold/weight ที่บันทึก
ไว้ใน thresholds.json/weights.json มาคำนวณ ensemble score แบบเดียวกับ
ตอนเทรน ก็จะได้ classification_report ชุดเดิมกลับมาโดยไม่ต้องรอ
GridSearchCV ใหม่ (ซึ่งกินเวลานานสุดในการเทรน)

Usage:
    python eval_trained_models.py --dataset-dir Dataset --model_dir model
"""

import argparse
import json
import os
import pickle
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hybrid_nids import (
    FEATURES, load_and_preprocess, DT_WEIGHT, RF_WEIGHT, XGB_WEIGHT,
)

_C17 = "CICIDS2017/MachineLearningCVE"
_C18 = "CICIDS2018/CSV"

ATTACK_FILE_CANDIDATES = {
    "PortScan": [f"{_C17}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
    "DoS": [f"{_C17}/Wednesday-workingHours.pcap_ISCX.csv"],
    "WebAttack": [
        f"{_C17}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        f"{_C18}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        f"{_C18}/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        f"{_C18}/02-22-2018.csv",
        f"{_C18}/02-23-2018.csv",
    ],
    "BruteForce": [
        f"{_C17}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        f"{_C18}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        f"{_C18}/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        f"{_C17}/Tuesday-WorkingHours.pcap_ISCX.csv",
        f"{_C18}/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="Dataset")
    p.add_argument("--model_dir", default="model")
    args = p.parse_args()

    thr_path = Path(args.model_dir, "thresholds.json")
    w_path = Path(args.model_dir, "weights.json")
    tuned_thresholds = json.load(open(thr_path)) if thr_path.exists() else {}
    trained_weights = json.load(open(w_path)) if w_path.exists() else {}

    attack_types = [("PortScan", "portscan"), ("DoS", "dos"),
                    ("WebAttack", "webattack"), ("BruteForce", "bruteforce")]

    for attack_label, attack_dir in attack_types:
        sub_dir = Path(args.model_dir, attack_dir)
        if not sub_dir.exists():
            print(f"[{attack_label}] ไม่พบโมเดลใน {sub_dir} — ข้าม")
            continue

        candidates = ATTACK_FILE_CANDIDATES.get(attack_label, [])
        found = [f for f in candidates
                 if os.path.exists(os.path.join(args.dataset_dir, f))]
        if not found:
            print(f"[{attack_label}] ไม่เจอไฟล์ dataset ต้นทางเลย ({candidates}) — ข้าม")
            continue

        print(f"\n{'='*60}\n{attack_label}  ←  {found}\n{'='*60}")

        import pandas as pd
        parts = [load_and_preprocess(os.path.join(args.dataset_dir, f)) for f in found]
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        df = df[df["Label"].isin(["Benign", attack_label])].copy()

        X_raw = df[FEATURES].values
        y_raw = (df["Label"] == attack_label).astype(int).values

        # ── reproduce split เดิมเป๊ะ (random_state=42 เหมือน train_models) ──
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_raw, y_raw, test_size=0.4, random_state=42, stratify=y_raw)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

        dt = pickle.load(open(sub_dir / "dt_model.pkl", "rb"))
        rf = pickle.load(open(sub_dir / "rf_model.pkl", "rb"))
        xgb = pickle.load(open(sub_dir / "xgb_model.pkl", "rb"))
        scaler = pickle.load(open(sub_dir / "scaler.pkl", "rb"))
        X_te_s = scaler.transform(X_te)

        w_dt, w_rf, w_xgb = trained_weights.get(
            attack_label, (DT_WEIGHT, RF_WEIGHT, XGB_WEIGHT))
        thr = tuned_thresholds.get(attack_label, 0.5)

        test_p = (w_dt * dt.predict_proba(X_te_s)[:, 1]
                  + w_rf * rf.predict_proba(X_te_s)[:, 1]
                  + w_xgb * xgb.predict_proba(X_te_s)[:, 1])
        test_pred = (test_p >= thr).astype(int)

        print(f"threshold={thr}  weights(DT,RF,XGB)={(w_dt, w_rf, w_xgb)}")
        print(f"Test set: {len(y_te):,} rows "
              f"({attack_label}={int(y_te.sum()):,} / Benign={int((y_te==0).sum()):,})")
        print(classification_report(
            y_te, test_pred, target_names=["Benign", attack_label],
            zero_division=0, digits=4))


if __name__ == "__main__":
    main()
