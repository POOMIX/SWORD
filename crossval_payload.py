#!/usr/bin/env python3
"""
crossval_payload.py — ตรวจ overfit ของ Payload ML (SQLi/XSS) ด้วย cross-validation

ทำไมต้องมีสคริปต์นี้:
  conf 1.000 ตอน inference "อาจ" หมายถึง 2 อย่าง
    (ก) overfit/ท่องจำ  → ไม่ดี ใช้จริงพัง
    (ข) payload แยกง่ายจริง (SQLi มี n-gram เฉพาะที่ปกติไม่มีเลย) → ปกติ
  ต้องพิสูจน์ว่าเป็น (ก) หรือ (ข) ด้วย cross-validation ที่ "ไม่โกง"

จุดเสี่ยงที่แท้จริงของ payload dataset = DUPLICATE LEAKAGE
  dataset เว็บ-payload ส่วนใหญ่มี payload "ซ้ำ/เกือบซ้ำ" เยอะมาก (เช่น
  <script>alert(1)</script> โผล่เป็นพัน) ถ้า random split เอา payload
  อันเดียวกันไปอยู่ทั้ง train และ test → โมเดลแค่ "ท่องจำ" ก็ได้ 1.000
  ทั้งที่ generalize ไม่ได้จริง → นี่คือ P1 (Sampling Bias) / spatial
  leakage ตาม Arp et al. USENIX 2022

สคริปต์นี้ทำ 4 อย่าง:
  1. วิเคราะห์ duplicate ในชุดข้อมูล (ต้นเหตุ overfit ที่พบบ่อยสุด)
  2. Stratified K-Fold CV (มาตรฐาน) — vectorizer fit ใหม่ทุก fold (ไม่ leak
     ข้าม fold) รายงาน per-class + train-vs-val gap (ตัวชี้ overfit)
  3. Grouped K-Fold CV — บังคับ payload ที่ "เหมือนกันเป๊ะ" อยู่ fold เดียว
     ไม่มีวันข้าม train/test → คะแนนนี้คือคะแนน "จริง" ที่ไม่ปนการท่องจำ
  4. เทียบ (2) vs (3): ถ้าตกเยอะ = เดิมสูงเพราะ duplicate leakage (overfit)
                       ถ้าตกนิดเดียว = payload แยกง่ายจริง (ไม่ overfit)

ใช้ hyperparameter ชุดเดียวกับ train_payload.py เป๊ะ (char_wb 2-4,
max_features=3000, XGBoost multiclass เดิม, class weight balanced)

วิธีใช้:
    python3 crossval_payload.py web_payloads_merged.csv
    python3 crossval_payload.py web_payloads_merged.csv --folds 5
"""
import argparse
import sys
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

LABEL_NAMES = {0: "Benign", 1: "SQL Injection", 2: "XSS"}


def load_csv(path):
    df = pd.read_csv(path)
    if "payload" not in df.columns or "label" not in df.columns:
        sys.exit(f"ไฟล์ต้องมีคอลัมน์ 'payload' และ 'label' (ที่เจอ: {list(df.columns)})")
    # preprocess เหมือน train_payload.py / hybrid_nids._run_payload_ml เป๊ะ
    df["payload"] = df["payload"].astype(str).apply(lambda x: unquote(x).lower())
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def make_vectorizer():
    # เหมือน train_payload.py บรรทัด vectorizer เป๊ะ
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                           max_features=3000, sublinear_tf=True)


def make_model():
    # เหมือน train_payload.py เป๊ะ
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────
# 1) วิเคราะห์ duplicate — ต้นเหตุ overfit ที่พบบ่อยสุดใน payload dataset
# ─────────────────────────────────────────────────────────────
def analyze_duplicates(df):
    print("\n" + "=" * 64)
    print(" 1) DUPLICATE ANALYSIS (ต้นเหตุ overfit ที่พบบ่อยสุด)")
    print("=" * 64)
    n = len(df)
    n_uniq = df["payload"].nunique()
    dup_rate = (n - n_uniq) / n * 100
    print(f"  ทั้งหมด        : {n:,} rows")
    print(f"  payload ไม่ซ้ำ  : {n_uniq:,} unique")
    print(f"  ซ้ำ            : {n - n_uniq:,} rows ({dup_rate:.1f}%)")
    print("  แยกตาม class (unique / total):")
    for c in sorted(df["label"].unique()):
        sub = df[df["label"] == c]["payload"]
        print(f"     {LABEL_NAMES.get(c, c):<14}: {sub.nunique():,} / {len(sub):,} "
              f"unique  (ซ้ำ {(len(sub)-sub.nunique())/max(len(sub),1)*100:.1f}%)")
    # payload ที่ซ้ำมากสุด
    top = df["payload"].value_counts().head(5)
    print("  payload ที่ซ้ำมากสุด 5 อันดับ:")
    for p, c in top.items():
        print(f"     {c:>6}×  {p[:60]!r}")
    if dup_rate > 20:
        print(f"\n  ⚠️  ซ้ำ {dup_rate:.0f}% — random-split CV จะสูงเกินจริงเพราะ leakage")
        print(f"      ให้ดู Grouped-CV (ข้อ 3) เป็นคะแนน 'จริง' แทน")
    else:
        print(f"\n  ✅ ซ้ำน้อย ({dup_rate:.0f}%) — random-split CV เชื่อถือได้พอควร")
    return dup_rate


# ─────────────────────────────────────────────────────────────
# helper: รัน CV loop 1 รอบ (ใช้ทั้ง stratified และ grouped)
# ─────────────────────────────────────────────────────────────
def run_cv(df, splitter, split_args, title):
    print("\n" + "=" * 64)
    print(f" {title}")
    print("=" * 64)
    X = df["payload"].values
    y = df["label"].values

    fold_val_f1, fold_train_f1, fold_acc = [], [], []
    all_true, all_pred = [], []

    for k, (tr, te) in enumerate(splitter.split(X, y, *split_args), 1):
        vec = make_vectorizer()
        Xtr = vec.fit_transform(X[tr])     # fit เฉพาะ train fold — กัน leak ข้าม fold
        Xte = vec.transform(X[te])
        sw = compute_sample_weight("balanced", y[tr])
        clf = make_model()
        clf.fit(Xtr, y[tr], sample_weight=sw)

        pred_te = clf.predict(Xte)
        pred_tr = clf.predict(Xtr)
        vf1 = f1_score(y[te], pred_te, average="macro", zero_division=0)
        tf1 = f1_score(y[tr], pred_tr, average="macro", zero_division=0)
        acc = (pred_te == y[te]).mean()
        fold_val_f1.append(vf1); fold_train_f1.append(tf1); fold_acc.append(acc)
        all_true.extend(y[te]); all_pred.extend(pred_te)
        gap = tf1 - vf1
        flag = "  ← gap สูง (overfit?)" if gap > 0.05 else ""
        print(f"  fold {k}: val macro-F1={vf1:.4f} | train={tf1:.4f} | "
              f"gap={gap:+.4f} | acc={acc:.4f}{flag}")

    vmean, vstd = np.mean(fold_val_f1), np.std(fold_val_f1)
    tmean = np.mean(fold_train_f1)
    print(f"\n  รวม {len(fold_val_f1)} folds:")
    print(f"     val macro-F1   = {vmean:.4f} ± {vstd:.4f}")
    print(f"     train macro-F1 = {tmean:.4f}")
    print(f"     train-val gap  = {tmean - vmean:+.4f}  "
          f"({'ใหญ่ = overfit' if tmean - vmean > 0.05 else 'เล็ก = ไม่ overfit'})")
    print(f"     accuracy       = {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")

    print("\n  รายงานรวมทุก fold (per-class):")
    labs = sorted(set(all_true) | set(all_pred))
    print(classification_report(all_true, all_pred, labels=labs,
                                target_names=[LABEL_NAMES.get(i, i) for i in labs],
                                zero_division=0, digits=4))
    print("  Confusion matrix (แถว=จริง, คอลัมน์=ทาย):")
    cm = confusion_matrix(all_true, all_pred, labels=labs)
    hdr = "        " + "".join(f"{LABEL_NAMES.get(i,i)[:8]:>10}" for i in labs)
    print(hdr)
    for i, row in zip(labs, cm):
        print(f"  {LABEL_NAMES.get(i,i)[:6]:<6}" + "".join(f"{v:>10,}" for v in row))
    return vmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ไฟล์ payload CSV (คอลัมน์ payload,label) — ตัวเดียวกับที่ train")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = load_csv(args.csv)
    print(f"โหลด {len(df):,} rows จาก {args.csv}")
    print("class distribution:")
    for c, n in df["label"].value_counts().sort_index().items():
        print(f"   {LABEL_NAMES.get(c,c):<14}: {n:,}")

    dup_rate = analyze_duplicates(df)

    # 2) Stratified K-Fold (มาตรฐาน — payload ซ้ำอาจข้าม train/test ได้)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    strat = run_cv(df, skf, (),
                   f"2) STRATIFIED {args.folds}-FOLD CV (มาตรฐาน)")

    # 3) Grouped K-Fold (payload เหมือนกันเป๊ะ = group เดียว ไม่ข้าม fold)
    groups = pd.factorize(df["payload"])[0]
    n_groups = len(set(groups))
    if n_groups < args.folds:
        print(f"\n[ข้าม Grouped-CV] payload unique {n_groups} < folds {args.folds}")
        grouped = None
    else:
        gkf = GroupKFold(n_splits=args.folds)
        grouped = run_cv(df, gkf, (groups,),
                         f"3) GROUPED {args.folds}-FOLD CV "
                         f"(payload ซ้ำไม่ข้าม train/test = คะแนนจริง)")

    # 4) สรุปคำตัดสิน
    print("\n" + "=" * 64)
    print(" 4) สรุป: overfit หรือไม่?")
    print("=" * 64)
    print(f"  Stratified CV macro-F1 : {strat:.4f}")
    if grouped is not None:
        drop = strat - grouped
        print(f"  Grouped CV    macro-F1 : {grouped:.4f}")
        print(f"  ตกลง                   : {drop:+.4f}")
        if drop > 0.05:
            print(f"\n  ⚠️  ตก {drop:.3f} เมื่อกัน duplicate leakage")
            print(f"      = คะแนนสูงเดิมส่วนหนึ่งมาจากการ 'ท่องจำ' payload ซ้ำ")
            print(f"      = มี overfit จาก duplicate → ควร dedup dataset ก่อนสรุปผล")
            print(f"      คะแนน 'จริง' ที่ควรรายงานในเล่ม = Grouped CV ({grouped:.4f})")
        else:
            print(f"\n  ✅ ตกแค่ {drop:.3f} — payload แยกได้จริง ไม่ได้ overfit จากการท่องจำ")
            print(f"      conf สูงมาจาก n-gram ที่เป็นสัญญาณจริง (SQLi/XSS แยกง่ายจริง)")
            print(f"      รายงานได้ทั้งสองตัวเลขในเล่มเพื่อความโปร่งใส")
    print("\n  หลักอ่านผล:")
    print("   - train-val gap เล็ก (<0.05) = โมเดลไม่ overfit ต่อ noise")
    print("   - Stratified≈Grouped         = ไม่ overfit ต่อ duplicate")
    print("   - ทั้งสองอย่างผ่าน + conf สูง = payload แยกง่ายจริง (ไม่ใช่ปัญหา)")


if __name__ == "__main__":
    main()
