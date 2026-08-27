"""
train_payload.py — Payload-based ML (XSS & SQLi)  v2
ใช้ TF-IDF (char n-gram) + XGBoost Multiclass
แยก Pipeline ออกจาก Flow-based ML

v2 changelog (แก้ตามหลักวิจัย Arp et al. USENIX Security 2022):
  - [FIX] เพิ่ม class weighting (sample_weight="balanced") — เดิม XGBoost
    ไม่ได้ balance class เลย ต่างจาก DT/RF ฝั่ง flow-model
  - [FIX] Train/Val/Test แบบไม่ leak (60/20/20) — val ใช้ tune confidence
    threshold, test ใช้รายงานผลจริงครั้งเดียว (เหมือนที่แก้ไว้ใน hybrid_nids.py)
  - [FIX] Confidence threshold ไม่ hardcode 0.85 แล้ว — tune จาก val set จริง
    โดย optimize F0.5 (เน้น precision ลด FP) แล้ว save เป็น payload_meta.json
    ให้ hybrid_nids.py โหลดไปใช้แทนค่าคงที่
  - [NEW] --ood-file รองรับ out-of-distribution test set (P9, Lab-Only Eval)
    รายงานแยกจาก in-distribution test เพื่อเช็ค shortcut learning ตาม
    adversarial-validation ที่ merge_payload_datasets.py เช็คไว้ก่อนหน้า

Dataset ที่ต้องการ: CSV มี column ['payload', 'label']
  label: 0=Benign, 1=SQL Injection, 2=Cross-Site Scripting (XSS)
  (ใช้ merge_payload_datasets.py สร้างไฟล์นี้จาก raw dataset หลายแหล่งได้)
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

LABEL_NAMES = {0: "Benign", 1: "SQL Injection", 2: "Cross-Site Scripting (XSS)"}


def _load_payload_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "payload" not in df.columns or "label" not in df.columns:
        raise KeyError(
            f"ไฟล์ {csv_path} ต้องมีคอลัมน์ 'payload' และ 'label' "
            f"(ที่มี: {list(df.columns)}) — ใช้ merge_payload_datasets.py สร้างไฟล์นี้ก่อน"
        )
    df["payload"] = df["payload"].astype(str).apply(lambda x: unquote(x).lower())
    df["label"] = df["label"].astype(int)
    return df


def _tune_confidence_threshold(model, X_val, y_val) -> tuple:
    """
    หา confidence threshold ที่ดีที่สุดสำหรับตัดสินใจ 'เป็น attack หรือไม่'
    โดย optimize F0.5 (เน้น precision มากกว่า recall 2 เท่า → ลด FP) บน
    validation set จริง แทนการ hardcode 0.85 แบบเดา (เหมือนหลักการเดียวกับ
    _auto_tune_threshold() ใน hybrid_nids.py)
    """
    probs = model.predict_proba(X_val)
    pred_class = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    y_val_binary = (y_val != 0).astype(int)

    thresholds = [round(x * 0.05, 2) for x in range(10, 20)]  # 0.50 - 0.95
    best_t, best_f = 0.85, -1.0
    rows = []
    for t in thresholds:
        pred_attack = ((pred_class != 0) & (conf >= t)).astype(int)
        f = fbeta_score(y_val_binary, pred_attack, beta=0.5, zero_division=0)
        rows.append((t, f))
        if f > best_f:
            best_t, best_f = t, f

    thr_str = "  ".join(f"t={t:.2f}→F0.5={f:.3f}{'★' if t == best_t else ''}" for t, f in rows)
    print(f"[Payload Threshold search]\n     {thr_str}")
    print(f"Best confidence threshold = {best_t:.2f}  (F0.5 = {best_f:.4f})")
    return best_t, best_f


def _report(model, vectorizer, threshold, X_text, y_true, title):
    Xv = vectorizer.transform(X_text)
    probs = model.predict_proba(Xv)
    pred_class = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    # ใช้ threshold แบบเดียวกับตอน inference จริง: ถ้า confidence ไม่ถึง
    # threshold ให้ถือว่าเป็น Benign แม้โมเดลจะ argmax เป็น attack ก็ตาม
    final_pred = np.where((pred_class != 0) & (conf >= threshold), pred_class, 0)

    print(f"\n{'═' * 60}")
    print(f" {title}")
    print(f"{'═' * 60}")
    labels_present = sorted(set(y_true) | set(final_pred))
    print(classification_report(
        y_true, final_pred,
        labels=labels_present,
        target_names=[LABEL_NAMES[i] for i in labels_present],
        zero_division=0, digits=4,
    ))
    return final_pred


def train_payload_model(csv_path: str, model_dir: str = "./model",
                         ood_path: str = None):
    print(" Train Payload-based ML (XSS & SQLi) — v2")
    t0 = time.time()

    df = _load_payload_csv(csv_path)
    print(f"Dataset: {len(df):,} rows")
    print(f"Class distribution:\n{df['label'].value_counts().rename(LABEL_NAMES).to_string()}")

    # ── Train / Val / Test = 60 / 20 / 20 (ไม่ leak) ─────────────────
    # val ใช้ tune confidence threshold, test รายงานผลจริงครั้งเดียวท้ายสุด
    df_tr, df_tmp = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df["label"])
    df_val, df_te = train_test_split(
        df_tmp, test_size=0.5, random_state=42, stratify=df_tmp["label"])
    print(f"Split → Train: {len(df_tr):,}  Val: {len(df_val):,}  Test: {len(df_te):,}")

    # ── TF-IDF Vectorizer (char_wb = character within word boundaries) ──
    print(" Creating TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=3000, sublinear_tf=True,
    )
    X_tr = vectorizer.fit_transform(df_tr["payload"])
    y_tr = df_tr["label"].values

    # ── Class weighting (แก้จาก v1 ที่ไม่ balance เลย) ────────────────
    sample_weight = compute_sample_weight("balanced", y_tr)
    print(f" Class weights (balanced) เฉลี่ยต่อ class: "
          f"{ {int(c): round(float(sample_weight[y_tr == c].mean()), 3) for c in np.unique(y_tr)} }")

    # ── XGBoost Multiclass ────────────────────────────────────────────
    print(" Training XGBoost (multiclass)...")
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr, sample_weight=sample_weight)

    # ── Tune confidence threshold บน Val ─────────────────────────────
    X_val = vectorizer.transform(df_val["payload"])
    threshold, val_f05 = _tune_confidence_threshold(xgb, X_val, df_val["label"].values)

    # ── รายงานผลจริงบน Test (in-distribution, ไม่เคยถูกแตะ) ──────────
    _report(xgb, vectorizer, threshold, df_te["payload"], df_te["label"].values,
            "In-distribution TEST report (held-out, ไม่เคยใช้ fit/tune)")

    # ── OOD report (out-of-distribution — P9 Lab-Only Eval check) ────
    if ood_path:
        df_ood = _load_payload_csv(ood_path)
        print(f"\nOOD dataset: {len(df_ood):,} rows "
              f"({Path(ood_path).name}, ไม่เคยเห็นระหว่าง train เลย)")
        _report(xgb, vectorizer, threshold, df_ood["payload"], df_ood["label"].values,
                "OUT-OF-DISTRIBUTION report (แหล่งข้อมูลที่โมเดลไม่เคยเห็น)")
        print(
            "\nหมายเหตุ: ถ้าตัวเลข OOD ต่ำกว่า In-distribution test มาก "
            "(เช่น F1 ต่างกันเกิน ~0.15-0.20) เป็นสัญญาณว่าโมเดลอาจเรียนรู้ "
            "shortcut/fingerprint ของ dataset แทน syntax การโจมตีจริง "
            "ควรกลับไปดู Adversarial Validation ตอน merge dataset อีกครั้ง"
        )

    # ── Save ──────────────────────────────────────────────────────────
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for name, obj in [("payload_vectorizer.pkl", vectorizer), ("payload_xgb.pkl", xgb)]:
        with open(Path(model_dir, name), "wb") as f:
            pickle.dump(obj, f)

    meta = {
        "threshold": threshold,
        "val_f0.5": val_f05,
        "label_names": LABEL_NAMES,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": len(df_tr), "val_rows": len(df_val), "test_rows": len(df_te),
    }
    with open(Path(model_dir, "payload_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n Done. Time: {time.time() - t0:.2f}s")
    print(f"Saved → {model_dir}/payload_vectorizer.pkl + payload_xgb.pkl + payload_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", nargs="?", default="web_payloads_merged.csv")
    p.add_argument("--model_dir", default="./model")
    p.add_argument("--ood-file", default=None,
                   help="ไฟล์ held-out out-of-distribution test (จาก merge_payload_datasets.py --ood-out)")
    args = p.parse_args()
    train_payload_model(args.csv_path, args.model_dir, args.ood_file)