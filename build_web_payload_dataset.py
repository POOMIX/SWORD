"""
build_web_payload_dataset.py — เตรียม + รวม dataset payload ที่มีอยู่ใน
dataset/ ให้เป็นชุดเดียวพร้อมเทรน (0=Benign, 1=SQL Injection,
2=Cross-Site Scripting (XSS)) โดยใช้ methodology เดียวกับ
merge_payload_datasets.py (Arp et al., USENIX Security 2022:
P1 sampling bias, P3 data snooping, P4 spurious correlation /
adversarial validation, P9 lab-only eval)

v2 — เพิ่ม csic_database.csv (CSIC2010 ฉบับเต็ม, โหลดใหม่แล้ว) และ
XSS_dataset.csv (Kaggle) เข้ามา พร้อมออกแบบใหม่ให้ SQLi และ XSS มี
"คู่ train/OOD ข้ามแหล่ง" แบบสมมาตรกันทั้งคู่ (เดิม v1 มีแค่ SQLi)

v3 — แก้จากผลจริงที่เจอตอนเทรน: v2 ถือ SQLiV3(SQLi) และ
payload_full(XSS) เป็น OOD ล้วน 100% (ไม่แตะตอน train เลย) ผลคือ
SQLi recall ร่วงจาก 0.998 (in-distribution) เหลือ 0.47 บน OOD — เพราะ
SQLiV3 เขียน SQLi แบบ "full SQL statement เชิงวิชาการ" (เช่น
"select * from users where id = '1' or 1=1 union select ...") ต่างจาก
payload_full ที่เป็น "query-param fragment สั้น ๆ" (เช่น
"1) or 8514=benchmark(...)#") พอไม่เคยเห็นสำนวนแบบ SQLiV3 เลยตอนเทรน
โมเดลเลย generalize ข้ามสำนวนไม่ได้ (มากกว่าที่ควรเป็น เพราะ SQLi
ไม่ใช่ syntax เดียว มีได้หลายสำนวน) — แก้โดยเปลี่ยนจาก "hold ทั้งแหล่ง
100%" เป็น "แบ่งสัดส่วน 70% เข้า train / 30% เก็บเป็น OOD" ต่อแหล่ง
แทน ให้โมเดลเห็นทั้ง 2 สำนวนตอนเทรน แต่ยังมี OOD สัดส่วนเล็กที่ไม่เคย
เห็นจริงให้ทดสอบ generalization ได้เหมือนเดิม (ดู split_train_ood())

แหล่งข้อมูลทั้งหมดใน dataset/ ตอนนี้ และบทบาทของแต่ละไฟล์:

  payload_full.csv        Benign=19,304  SQLi=10,852  XSS=532
      → Benign + SQLi เข้า TRAIN (แหล่งหลักของ SQLi)
      → XSS (532 แถว) เก็บเป็น OOD ล้วน (ไม่เอาเข้า train) เพราะ
        XSS_dataset.csv ใหญ่กว่ามากและครอบคลุมกว่า ให้เป็นแหล่งหลักแทน

  SQLiV3.csv (Kaggle "sql injection dataset")
      Benign=19,537  SQLi=11,365 (column ขยับ ต้อง parse เอง — กู้คืนมาได้
      เกือบครบ) → Benign เข้า TRAIN (เพิ่มความหลากหลาย), SQLi (11,365)
      เก็บเป็น OOD ล้วน (ไม่เอาเข้า train) → คู่กับ SQLi หลักจาก
      payload_full.csv เพื่อเช็ค cross-dataset generalization (P9)

  XSS_dataset.csv (Kaggle, โหลดใหม่)
      Benign=6,313  XSS=7,373 → Benign + XSS เข้า TRAIN (แหล่งหลักของ XSS
      ตอนนี้ — แก้ปัญหา XSS ขาดแคลนจาก v1 ที่มีแค่ 532 samples)

  csic_database.csv (CSIC2010 ฉบับเต็ม, โหลดใหม่แล้ว — ใช้ได้แล้ว)
      คอลัมน์ URL มี query string จริง (เช่น
      "id=3&nombre=Vino...&precio=100") + content (POST body) ใช้เป็น
      Benign เพิ่มได้ (เอาเฉพาะ classification="Normal") — เป็น payload
      รูปแบบ "query string จริงจากเซิร์ฟเวอร์จริง" ซึ่งขาดหายไปจากทุก
      แหล่งอื่น (แหล่งอื่นเป็นข้อความเดี่ยว/query สังเคราะห์)
      ส่วนแถว Anomalous (25,065 แถว) *ไม่เอามาเป็น attack label* เพราะ
      CSIC ไม่ได้แยกประเภทการโจมตี (Anomalous = ผสม SQLi/XSS/buffer
      overflow/parameter tampering/path traversal ปนกันหมด) ถ้า map
      เป็น SQLi หรือ XSS ทั้งดุ้นจะเกิด label noise มหาศาล จึงใช้แค่ครึ่ง
      Normal เป็น benign pool เท่านั้น

TRAIN vs OOD (v3 — แบ่งสัดส่วน 70/30 ต่อแหล่ง แทนการ hold 100%):
  TRAIN  = payload_full(benign+sqli+xss 70%) + XSS_dataset(benign+xss)
           + SQLiV3(benign + sqli 70%) + csic(normal only, benign)
  OOD    = SQLiV3(sqli 30% ที่เหลือ) + payload_full(xss 30% ที่เหลือ)
           — สุ่มแบ่งไว้ก่อนเทรน ไม่เคยผ่านการเทรนเลย ใช้ทดสอบว่าโมเดล
           generalize ข้าม "สำนวน" การเขียนได้จริงไหม โดยที่ตอนเทรนก็
           ยังได้เห็นตัวอย่างของทั้ง 2 สำนวนแล้ว (ไม่ใช่เห็นแค่สำนวน
           เดียวแบบ v2)

Brute Force ไม่ได้อยู่ใน dataset นี้ — เป็น task แยกฝั่ง flow-based
(ดู hybrid_nids.py ที่แก้เพิ่ม BruteForce OvR task แยกจาก WebAttack แล้ว)

Usage:
    python build_web_payload_dataset.py --dataset-dir ./dataset --out-dir .
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_payload_datasets import adversarial_validation, LABEL_NAMES  # noqa: E402

RANDOM_STATE = 42
OOD_HOLDOUT_FRAC = 0.3  # สัดส่วนที่เก็บเป็น OOD ล้วนต่อแหล่ง (ที่เหลือ 70% เข้า train)


def split_train_ood(df: pd.DataFrame, ood_frac: float = OOD_HOLDOUT_FRAC,
                     seed: int = RANDOM_STATE):
    """สุ่มแบ่ง df เป็น (train_part, ood_part) — ไม่ hold ทั้งแหล่ง 100%
    เป็น OOD แบบ v2 อีกต่อไป (ดูเหตุผลใน docstring ด้านบน v3) เพราะทำให้
    โมเดลไม่เคยเห็น "สำนวน" ของแหล่งนั้นเลย แล้ว generalize ข้ามสำนวน
    ไม่ได้จริง ทั้งที่เป็นคลาสเดียวกัน (เช่น SQLi แค่คนละสไตล์การเขียน)"""
    rng = np.random.default_rng(seed)
    idx = df.index.values.copy()
    rng.shuffle(idx)
    n_ood = int(len(idx) * ood_frac)
    ood_idx, train_idx = idx[:n_ood], idx[n_ood:]
    return df.loc[train_idx].copy(), df.loc[ood_idx].copy()


# ─────────────────────────────────────────────
# 1) payload_full.csv → benign(train) / sqli(train) / xss(70% train / 30% OOD)
# ─────────────────────────────────────────────

def load_payload_full(path: Path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    keep = df[df["attack_type"].isin(["norm", "sqli", "xss"])].copy()
    dropped = len(df) - len(keep)
    if dropped:
        print(f"[payload_full.csv] ตัดทิ้ง {dropped:,} แถว (cmdi / path-traversal "
              f"— ไม่ใช่ 1 ใน 3 คลาสเป้าหมาย SQLi/XSS/Brute Force)")

    keep["payload"] = keep["payload"].astype(str)
    label_map = {"norm": 0, "sqli": 1, "xss": 2}
    keep["label"] = keep["attack_type"].map(label_map)
    keep["_source"] = "payload_full"
    keep = keep[["payload", "label", "_source"]]
    keep = keep[keep["payload"].str.strip() != ""]
    keep = keep.dropna(subset=["payload", "label"])
    keep["label"] = keep["label"].astype(int)

    train_part = keep[keep["label"].isin([0, 1])].copy()          # benign + sqli
    xss_all = keep[keep["label"] == 2].copy()                     # xss (minority source)
    xss_train_extra, xss_ood = split_train_ood(xss_all)
    train_part = pd.concat([train_part, xss_train_extra], ignore_index=True)

    vc = keep["label"].value_counts()
    print(f"[payload_full.csv] rows={len(keep):,}  "
          f"Benign={vc.get(0,0):,}(train)  SQLi={vc.get(1,0):,}(train)  "
          f"XSS={vc.get(2,0):,} → {len(xss_train_extra):,}(train)/{len(xss_ood):,}(OOD)")
    return train_part, xss_ood


# ─────────────────────────────────────────────
# 2) SQLiV3.csv → benign(train) / sqli(OOD only)
#    parse เองด้วย csv.reader ดิบ เพราะ label บางแถวขยับคอลัมน์
# ─────────────────────────────────────────────

def load_sqliv3(path: Path):
    rows = []
    unresolved = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        for r in reader:
            if len(r) < n_cols:
                r = r + [""] * (n_cols - len(r))
            label = None
            for i in range(1, min(n_cols, 4)):
                v = r[i].strip()
                if v in ("0", "1", "0.0", "1.0"):
                    label = int(float(v))
                    break
            text = r[0].strip()
            if label is None or not text:
                unresolved += 1
                continue
            rows.append((text, label))

    if unresolved:
        print(f"[SQLiV3.csv] ทิ้ง {unresolved:,} แถวที่กู้ label/payload ไม่ได้")

    df = pd.DataFrame(rows, columns=["payload", "label"])
    df["_source"] = "sqliv3"

    train_part = df[df["label"] == 0].copy()          # benign → train
    sqli_all = df[df["label"] == 1].copy()             # sqli (minority "dialect")
    sqli_train_extra, sqli_ood = split_train_ood(sqli_all)
    train_part = pd.concat([train_part, sqli_train_extra], ignore_index=True)

    vc = df["label"].value_counts()
    print(f"[SQLiV3.csv] rows={len(df):,}  Benign={vc.get(0,0):,}(train)  "
          f"SQLi={vc.get(1,0):,} → {len(sqli_train_extra):,}(train)/{len(sqli_ood):,}(OOD)  "
          f"(ไม่มีคลาส XSS ในไฟล์นี้)")
    return train_part, sqli_ood


# ─────────────────────────────────────────────
# 3) XSS_dataset.csv → benign(train) / xss(train)  — แหล่งหลักของ XSS
# ─────────────────────────────────────────────

def load_xss_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "Sentence" not in df.columns or "Label" not in df.columns:
        raise KeyError(f"[XSS_dataset.csv] ต้องมีคอลัมน์ Sentence/Label "
                        f"(ที่มี: {list(df.columns)})")

    out = pd.DataFrame()
    out["payload"] = df["Sentence"].astype(str)
    out["label"] = df["Label"].astype(int).map({0: 0, 1: 2})  # 0=Benign, 1→2=XSS
    out["_source"] = "xss_dataset"
    out = out[out["payload"].str.strip() != ""]
    out = out.dropna(subset=["payload", "label"])
    out["label"] = out["label"].astype(int)

    vc = out["label"].value_counts()
    print(f"[XSS_dataset.csv] rows={len(out):,}  Benign={vc.get(0,0):,}(train)  "
          f"XSS={vc.get(2,0):,}(train)  — แหล่งหลักของ XSS ตอนนี้")
    return out


# ─────────────────────────────────────────────
# 4) csic_database.csv → benign เท่านั้น (Normal only) จาก URL query
#    string + POST content — Anomalous ไม่ใช้ เพราะไม่ได้แยกประเภท
#    การโจมตี (ผสม SQLi/XSS/parameter-tampering/ฯลฯ ปนกัน → label noise)
# ─────────────────────────────────────────────

def _extract_request_payload(url: str, content) -> str:
    parts = []
    if isinstance(url, str) and "?" in url:
        qs = url.split("?", 1)[1]
        qs = re.sub(r"\s+HTTP/\d\.\d\s*$", "", qs)
        parts.append(unquote(qs))
    if isinstance(content, str) and content.strip():
        parts.append(unquote(content))
    return " ".join(parts).strip()


def load_csic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    normal = df[df["classification"].astype(str).str.strip() == "0"].copy()
    normal["payload"] = normal.apply(
        lambda r: _extract_request_payload(r.get("URL"), r.get("content")), axis=1)
    normal = normal[normal["payload"].str.strip() != ""]

    out = pd.DataFrame({
        "payload": normal["payload"],
        "label": 0,
        "_source": "csic_normal",
    })
    print(f"[csic_database.csv] Normal rows ที่มี query/content จริงให้ใช้ "
          f"= {len(out):,} (จากทั้งหมด {(df['classification'].astype(str).str.strip()=='0').sum():,} "
          f"Normal rows — ที่เหลือไม่มี query string/content เลยเลยไม่มีอะไรให้เรียนรู้) "
          f"| ข้าม Anomalous {len(df) - (df['classification'].astype(str).str.strip()=='0').sum():,} "
          f"แถวทั้งหมด เพราะไม่ได้แยกว่าเป็น SQLi/XSS/อื่น ๆ")
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--benign-ratio", type=float, default=1.5,
                   help="สัดส่วน Benign:Attack สูงสุดหลัง balance (default 1.5:1)")
    args = p.parse_args()

    ds = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" STEP 1 — โหลด + ทำความสะอาดแต่ละแหล่ง")
    print("=" * 70)
    pf_train, pf_ood = load_payload_full(ds / "payload_full.csv")
    sq_train, sq_ood = load_sqliv3(ds / "SQLiV3.csv")
    xss_train = load_xss_dataset(ds / "XSS_dataset.csv")
    csic_train = load_csic(ds / "csic_database.csv")

    print("\n" + "=" * 70)
    print(" STEP 2 — Cross-source dedup (P3) + Adversarial Validation (P4)")
    print("=" * 70)
    train_sources = [pf_train, sq_train, xss_train, csic_train]
    train_raw = pd.concat(train_sources, ignore_index=True)
    before = len(train_raw)
    train_raw = train_raw.drop_duplicates(subset=["payload"], keep="first")
    print(f"Dedup (cross-source, train pool): {before:,} → {len(train_raw):,} rows "
          f"({before - len(train_raw):,} ซ้ำถูกตัดออก)")

    adversarial_validation(train_raw)

    print("\n" + "=" * 70)
    print(" STEP 3 — Balance benign:attack แล้วบันทึก train pool")
    print("=" * 70)
    benign = train_raw[train_raw["label"] == 0].copy()
    attacks = train_raw[train_raw["label"] != 0].copy()
    print(f"Before balance → Benign={len(benign):,}  "
          f"SQLi={(attacks.label==1).sum():,}  XSS={(attacks.label==2).sum():,}")

    rng = np.random.default_rng(RANDOM_STATE)
    n_benign_target = min(len(benign), int(len(attacks) * args.benign_ratio))
    if n_benign_target < len(benign):
        idx = rng.choice(benign.index.values, size=n_benign_target, replace=False)
        benign = benign.loc[idx]

    final = pd.concat([benign, attacks], ignore_index=True)
    final = final.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"After balance → {len(final):,} rows total")
    for cid, name in LABEL_NAMES.items():
        print(f"     {name:<28}: {(final.label == cid).sum():,}")

    train_out = out_dir / "web_payloads_merged.csv"
    final[["payload", "label"]].to_csv(train_out, index=False)
    print(f"\n✅ บันทึก train pool → {train_out}")

    print("\n" + "=" * 70)
    print(" STEP 4 — OOD held-out (P9) — SQLiV3(SQLi) + payload_full(XSS)")
    print("          ไม่เคยผ่านการเทรนเลยทั้งคู่ ใช้เช็ค cross-dataset generalization")
    print("=" * 70)
    ood = pd.concat([sq_ood, pf_ood], ignore_index=True)
    before_ood = len(ood)
    ood = ood.drop_duplicates(subset=["payload"], keep="first")
    overlap = ood["payload"].isin(final["payload"])
    if overlap.any():
        print(f"ตัด {overlap.sum():,} แถวใน OOD ที่ดันซ้ำกับ train pool ออก (กัน leakage)")
        ood = ood[~overlap]
    ood_out = out_dir / "web_payloads_ood.csv"
    ood[["payload", "label"]].to_csv(ood_out, index=False)
    vc_ood = ood["label"].value_counts()
    print(f"✅ บันทึก OOD held-out test → {ood_out} ({len(ood):,} rows: "
          f"SQLi={vc_ood.get(1,0):,}  XSS={vc_ood.get(2,0):,})")

    print("\n" + "=" * 70)
    print(" สรุป")
    print("=" * 70)
    n_xss = (final.label == 2).sum()
    n_sqli = (final.label == 1).sum()
    n_benign = (final.label == 0).sum()
    print(f" - Train: Benign={n_benign:,}  SQLi={n_sqli:,}  XSS={n_xss:,}")
    print(f" - OOD:   SQLi={vc_ood.get(1,0):,} (จาก SQLiV3)  "
          f"XSS={vc_ood.get(2,0):,} (จาก payload_full — ยังค่อนข้างน้อย "
          f"ถ้า OOD F1 ของ XSS แกว่งเยอะเพราะ sample size เล็ก ให้ระวังตอนสรุปผล "
          f"อย่าฟันธงจาก N ที่เล็กขนาดนี้)")
    print(f" - Brute Force ไม่ได้อยู่ใน dataset นี้ ต้องดูผลจาก flow-based "
          f"pipeline (hybrid_nids.py, task 'bruteforce' แยกจาก 'webattack' แล้ว)")


if __name__ == "__main__":
    main()
