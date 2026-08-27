"""
merge_payload_datasets.py — รวม dataset payload จากหลายแหล่งให้เป็น scheme
เดียว (0=Benign, 1=SQL Injection, 2=Cross-Site Scripting (XSS)) พร้อมจัดการ
ปัญหาเชิงระเบียบวิธีตาม Arp et al., "Dos and Don'ts of Machine Learning in
Computer Security" (USENIX Security 2022):

  - P1 Sampling Bias    → ผสม Benign pool จากทุกแหล่งแทนที่จะผูกกับ attack
                          type เดียว (กัน "benign แบบ A ผูกกับ attack แบบ A")
  - P3 Data Snooping    → Cross-source deduplication ก่อน split train/test
  - P4 Spurious Correl. → Adversarial Validation เช็คว่าโมเดลแยก "แหล่งข้อมูล"
                          ได้ง่ายเกินไปมั้ย (สัญญาณของ dataset fingerprint)
  - P9 Lab-Only Eval    → รองรับ --ood-file แยกไว้เป็น held-out ที่ไม่ถูกรวม
                          เข้า train เลย สำหรับวัด out-of-distribution จริง

ตัวอย่างการใช้งาน:
    python merge_payload_datasets.py \\
        --sqli-file sql_injection_dataset.csv \\
        --xss-file xss_dataset_for_deep_learning.csv \\
        --ood-file fmereani_xss_heldout.csv --ood-type xss \\
        --out web_payloads_merged.csv --ood-out web_payloads_ood.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = {0: "Benign", 1: "SQL Injection", 2: "Cross-Site Scripting (XSS)"}

TEXT_COL_CANDIDATES = [
    "payload", "Payload", "Payloads", "Sentence", "sentence",
    "Query", "query", "text", "Text", "url", "URL", "request", "Request",
]
LABEL_COL_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]

BENIGN_KEYWORDS = {"benign", "normal", "0", "0.0", "legit", "legitimate",
                    "plain", "plaintext", "plain text", "clean"}
ATTACK_KEYWORDS = {"malicious", "injection", "xss", "sqli", "attack", "1", "1.0",
                    "sql injection", "cross-site scripting", "sql_injection"}


# ─────────────────────────────────────────────
# Column / label auto-detection
# ─────────────────────────────────────────────

def _detect_column(df: pd.DataFrame, candidates: list, role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if role == "text":
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if not str_cols:
            raise ValueError(
                f"หา text column อัตโนมัติไม่เจอ | columns={list(df.columns)} "
                f"กรุณาระบุ --text-col เอง"
            )
        # heuristic: text column มักมีความยาวเฉลี่ยมากกว่า label column
        lengths = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        return max(lengths, key=lengths.get)
    raise ValueError(
        f"หา {role} column อัตโนมัติไม่เจอ | columns={list(df.columns)} "
        f"กรุณาระบุ --label-col เอง"
    )


def _normalize_binary_label(series: pd.Series) -> pd.Series:
    """แปลง label column (0/1 ตัวเลข หรือ string เช่น 'Malicious'/'Benign')
    ให้เป็น 0=benign, 1=attack เสมอ"""
    vals = series.astype(str).str.strip().str.lower()
    uniq = set(vals.unique())

    if uniq <= {"0", "1", "0.0", "1.0"}:
        return vals.map(lambda v: 1 if v in ("1", "1.0") else 0).astype(int)

    mapped = pd.Series(np.nan, index=series.index, dtype="float64")
    unresolved = []
    for v in uniq:
        is_benign = any(k in v for k in BENIGN_KEYWORDS)
        is_attack = any(k in v for k in ATTACK_KEYWORDS)
        if is_benign and not is_attack:
            mapped[vals == v] = 0
        elif is_attack and not is_benign:
            mapped[vals == v] = 1
        else:
            unresolved.append(v)
    if unresolved:
        raise ValueError(
            f"แปลง label ไม่ได้ ค่าที่ไม่รู้จัก: {unresolved} "
            f"กรุณาระบุ --positive-values เอง (comma-separated)"
        )
    return mapped.astype(int)


def load_source(path: str, attack_type: str, text_col=None, label_col=None,
                 positive_values=None) -> pd.DataFrame:
    """
    attack_type: 'sqli' หรือ 'xss' — บอกว่า label=1 (attack) ในไฟล์นี้ควร map
    ไปเป็น class ไหนใน scheme รวม (1=SQLi, 2=XSS)
    """
    assert attack_type in ("sqli", "xss"), "attack_type ต้องเป็น 'sqli' หรือ 'xss'"

    df = None
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python")
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if df is None:
        raise ValueError(f"อ่านไฟล์ {path} ไม่ได้ (ลอง utf-8/latin-1 แล้ว): {last_err}")

    df.columns = df.columns.astype(str).str.strip()
    tcol = text_col or _detect_column(df, TEXT_COL_CANDIDATES, "text")
    lcol = label_col or _detect_column(df, LABEL_COL_CANDIDATES, "label")
    print(f"[{Path(path).name}] rows={len(df):,}  text_col='{tcol}'  label_col='{lcol}'")

    out = pd.DataFrame()
    out["payload"] = df[tcol].astype(str)

    if positive_values:
        pos_set = {v.strip().lower() for v in positive_values.split(",")}
        binary = df[lcol].astype(str).str.strip().str.lower().isin(pos_set).astype(int)
    else:
        binary = _normalize_binary_label(df[lcol])

    class_id = 1 if attack_type == "sqli" else 2
    out["label"] = binary.map({0: 0, 1: class_id})
    out["_source"] = f"{Path(path).stem}:{attack_type}"
    out = out.dropna(subset=["label", "payload"])
    out = out[out["payload"].str.strip() != ""]
    out["label"] = out["label"].astype(int)

    vc = out["label"].value_counts()
    print(f"     → {LABEL_NAMES[0]}={vc.get(0,0):,}  {LABEL_NAMES[class_id]}={vc.get(class_id,0):,}")
    return out


# ─────────────────────────────────────────────
# P4 — Adversarial Validation (dataset fingerprint check)
# ─────────────────────────────────────────────

def adversarial_validation(df: pd.DataFrame) -> float:
    """
    เช็คว่า Benign sample จากแต่ละแหล่งข้อมูล "แยกออกจากกันได้ง่ายเกินไป" มั้ย
    ถ้าโมเดลง่ายๆ (Logistic Regression บน TF-IDF) แยกแหล่งที่มาได้แม่นสูงลิบ
    ทั้งที่เป็น Benign เหมือนกันหมด → สัญญาณว่า dataset มี "ลายเซ็น" ที่ไม่
    เกี่ยวกับการโจมตีเลย (สไตล์การเขียน/ความยาว/whitespace) ซึ่งโมเดลหลัก
    อาจแอบใช้เป็น shortcut แทนการเรียนรู้ syntax ของการโจมตีจริง
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    benign = df[df["label"] == 0]
    sources = benign["_source"].unique()
    print(f"\n[Adversarial Validation] Benign มาจาก {len(sources)} แหล่ง: {list(sources)}")

    if len(sources) < 2:
        print("     มีแหล่งเดียว — ข้ามการเช็ค (ไม่มีอะไรให้เปรียบเทียบ)")
        return 0.5

    # ใช้ 2 แหล่งที่มี sample เยอะที่สุดมาเทียบกัน (กรณีมีมากกว่า 2 แหล่ง)
    top2 = benign["_source"].value_counts().index[:2]
    sub = benign[benign["_source"].isin(top2)]
    y = (sub["_source"] == top2[0]).astype(int)
    if y.nunique() < 2 or min(y.sum(), len(y) - y.sum()) < 10:
        print("     แหล่งใดแหล่งหนึ่งมีตัวอย่างน้อยเกินไป — ข้ามการเช็ค")
        return 0.5

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=2000)
    X = vec.fit_transform(sub["payload"])
    clf = LogisticRegression(max_iter=1000)
    cv = min(5, int(min(y.sum(), len(y) - y.sum())))
    scores = cross_val_score(clf, X, y, cv=max(cv, 2), scoring="accuracy")
    acc = scores.mean()
    print(f"     แยกแหล่งที่มา '{top2[0]}' vs '{top2[1]}' ได้แม่น {acc:.1%} "
          f"(baseline ~50% ถ้าไม่มี fingerprint)")
    if acc > 0.75:
        print("     ⚠️  WARNING: dataset มี source-fingerprint สูง — โมเดลหลัก "
              "เสี่ยงเรียนรู้ shortcut (สไตล์เขียน/ความยาว) แทน syntax จริง "
              "ควร normalize text เพิ่ม (ตัด HTML entity, unify whitespace, "
              "ตัดคำ/URL เฉพาะโดเมนออก) ก่อนใช้เทรนจริง")
    else:
        print("     ✅ ดูสมเหตุสมผล ไม่มี fingerprint ชัดเจนเกินไป")
    return acc


# ─────────────────────────────────────────────
# Merge + dedup + balance
# ─────────────────────────────────────────────

def merge_and_balance(sources: list, benign_ratio: float = 1.5,
                       random_state: int = 42) -> pd.DataFrame:
    df = pd.concat(sources, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["payload"], keep="first")
    print(f"\nDedup (cross-source): {before:,} → {len(df):,} rows "
          f"({before - len(df):,} ซ้ำถูกตัดออก)")

    benign = df[df["label"] == 0].copy()
    attacks = df[df["label"] != 0].copy()
    print(f"Before balance → Benign={len(benign):,}  "
          f"SQLi={(attacks.label==1).sum():,}  XSS={(attacks.label==2).sum():,}")

    rng = np.random.default_rng(random_state)
    n_benign_target = min(len(benign), int(len(attacks) * benign_ratio))
    if n_benign_target < len(benign):
        idx = rng.choice(benign.index.values, size=n_benign_target, replace=False)
        benign = benign.loc[idx]

    out = pd.concat([benign, attacks], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"After balance → {len(out):,} rows total")
    for cid, name in LABEL_NAMES.items():
        print(f"     {name:<28}: {(out.label == cid).sum():,}")
    return out


def main():
    p = argparse.ArgumentParser(
        description="รวม dataset payload หลายแหล่งเป็นชุดเดียว (0=Benign,1=SQLi,2=XSS)")
    p.add_argument("--sqli-file", action="append", default=[],
                    help="ไฟล์ dataset SQLi (ใส่ได้หลายครั้ง)")
    p.add_argument("--xss-file", action="append", default=[],
                    help="ไฟล์ dataset XSS (ใส่ได้หลายครั้ง)")
    p.add_argument("--ood-file", action="append", default=[],
                    help="ไฟล์ held-out สำหรับ out-of-distribution test (ไม่ถูกรวมเข้า train)")
    p.add_argument("--ood-type", default="xss", choices=["sqli", "xss"])
    p.add_argument("--benign-ratio", type=float, default=1.5,
                    help="สัดส่วน Benign:Attack สูงสุดหลัง balance (default 1.5:1)")
    p.add_argument("--out", default="web_payloads_merged.csv")
    p.add_argument("--ood-out", default="web_payloads_ood.csv")
    args = p.parse_args()

    if not args.sqli_file and not args.xss_file:
        print("ต้องระบุอย่างน้อย 1 ไฟล์ --sqli-file หรือ --xss-file")
        sys.exit(1)

    sources = []
    for f in args.sqli_file:
        sources.append(load_source(f, "sqli"))
    for f in args.xss_file:
        sources.append(load_source(f, "xss"))

    merged_raw = pd.concat(sources, ignore_index=True)
    adversarial_validation(merged_raw)

    final = merge_and_balance(sources, benign_ratio=args.benign_ratio)
    final[["payload", "label"]].to_csv(args.out, index=False)
    print(f"\n✅ บันทึก train pool → {args.out}")

    if args.ood_file:
        ood_sources = [load_source(f, args.ood_type) for f in args.ood_file]
        ood_df = pd.concat(ood_sources, ignore_index=True).drop_duplicates(subset=["payload"])
        overlap = ood_df["payload"].isin(final["payload"])
        if overlap.any():
            print(f"\nตัด {overlap.sum():,} แถวใน OOD ที่ซ้ำกับ train pool ออก (กัน leakage)")
            ood_df = ood_df[~overlap]
        ood_df[["payload", "label"]].to_csv(args.ood_out, index=False)
        print(f"✅ บันทึก OOD held-out test → {args.ood_out} ({len(ood_df):,} rows)")


if __name__ == "__main__":
    main()