#!/usr/bin/env python3
"""
check_cicids_flags.py — ตรวจ "ค่าดิบ" ในไฟล์ CICIDS โดยไม่ผ่าน preprocessing

ทำไมต้องมี: compare_train_live.py บอกว่า flow ที่ label เป็น PortScan ใน
ชุดเทรนมีค่ากลาง SYN=0, RST=0, ACK=0, PSH=1 ซึ่ง "เป็นไปไม่ได้" ทาง
เทคนิค เพราะ SYN scan ต้องส่ง SYN เสมอตามนิยาม และไม่มีทางมี PSH
(PSH = มี application data ส่งจริง ซึ่ง scan ไม่เคยทำ)

มีสองความเป็นไปได้ ต้องแยกให้ออกก่อนแก้ ไม่งั้นแก้ผิดจุดแล้วเสียเวลา
retrain ฟรีอีกรอบ:

  (ก) โค้ดเรา map คอลัมน์ผิด  -> เป็นบั๊กของเรา แก้ได้ทันที
  (ข) ค่าในไฟล์เป็นแบบนั้นจริง -> เป็นข้อจำกัดของ CICFlowMeter/dataset
                                 ต้องตัด feature กลุ่มนั้นออก

สคริปต์นี้อ่าน CSV ดิบ ๆ ไม่ผ่าน rename/preprocess ใด ๆ แล้วโชว์:
  - ชื่อคอลัมน์จริงทุกตัวที่มีคำว่า Flag / Length / Packets
  - การกระจายค่าของคอลัมน์ flag แยกตาม label
  - ค่า byte columns แยกตาม label
เทียบ PortScan กับ BENIGN ในไฟล์เดียวกัน

ใช้งาน:
  python3 check_cicids_flags.py \\
      --csv Dataset/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \\
      --label PortScan
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="PortScan",
                    help="ชื่อ label ดิบในไฟล์ (เช่น PortScan, DoS Hulk, FTP-Patator)")
    ap.add_argument("--rows", type=int, default=400000,
                    help="อ่านกี่แถว (default 400000 พอสำหรับดูการกระจาย)")
    args = ap.parse_args()

    print(f"อ่านดิบ ๆ: {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False, nrows=args.rows)
    df.columns = df.columns.str.strip()
    print(f"   {len(df):,} แถว, {len(df.columns)} คอลัมน์\n")

    # หาคอลัมน์ Label
    lab_col = next((c for c in df.columns if c.lower() == "label"), None)
    if not lab_col:
        print("ไม่พบคอลัมน์ Label")
        return
    labels = df[lab_col].astype(str).str.strip()
    print(f"label ที่มีในไฟล์: {sorted(labels.unique())}\n")

    mask_atk = labels == args.label
    mask_ben = labels.str.upper() == "BENIGN"
    print(f"{args.label}: {mask_atk.sum():,} แถว | BENIGN: {mask_ben.sum():,} แถว\n")
    if not mask_atk.any():
        print(f"ไม่พบ label '{args.label}' — ใช้ชื่อจากรายการด้านบน")
        return

    # ── คอลัมน์ flag ──────────────────────────────────────────────
    flag_cols = [c for c in df.columns if "flag" in c.lower()]
    print("=" * 78)
    print("คอลัมน์ FLAG ที่มีจริงในไฟล์ (ชื่อเป๊ะ ๆ)")
    print("=" * 78)
    for c in flag_cols:
        print(f"    {c!r}")
    print()

    print("=" * 78)
    print(f"ค่าของคอลัมน์ flag: {args.label} vs BENIGN")
    print("=" * 78)
    print(f"{'column':26s} {args.label[:12]:>12s}{'':4s}{'BENIGN':>12s}   หมายเหตุ")
    print("-" * 78)
    for c in flag_cols:
        a = pd.to_numeric(df.loc[mask_atk, c], errors="coerce")
        b = pd.to_numeric(df.loc[mask_ben, c], errors="coerce")
        a_pct = 100.0 * (a > 0).mean() if len(a) else 0
        b_pct = 100.0 * (b > 0).mean() if len(b) else 0
        note = ""
        if "syn" in c.lower() and a_pct < 50:
            note = "  <-- ผิดปกติ! SYN scan ต้องมี SYN เสมอ"
        if "psh" in c.lower() and a_pct > 50:
            note = "  <-- ผิดปกติ! scan ไม่ส่ง data จึงไม่ควรมี PSH"
        print(f"{c:26s} {a_pct:11.1f}% {'':4s}{b_pct:11.1f}%   {note}")
    print("\n  (ตัวเลข = % ของแถวที่คอลัมน์นั้นมีค่ามากกว่า 0)")
    print()

    # ── คอลัมน์ byte / packet ─────────────────────────────────────
    byte_cols = [c for c in df.columns
                 if ("length" in c.lower() or "len" in c.lower()
                     or "byts" in c.lower() or "bytes" in c.lower())
                 and "packet" in c.lower() or c.strip() in
                 ("Total Length of Fwd Packets", "Total Length of Bwd Packets",
                  "TotLen Fwd Pkts", "TotLen Bwd Pkts")]
    byte_cols = list(dict.fromkeys(byte_cols))
    if byte_cols:
        print("=" * 78)
        print(f"คอลัมน์ BYTE: {args.label} vs BENIGN (median)")
        print("=" * 78)
        print(f"{'column':34s} {args.label[:10]:>12s} {'BENIGN':>12s}   หมายเหตุ")
        print("-" * 78)
        for c in byte_cols:
            a = pd.to_numeric(df.loc[mask_atk, c], errors="coerce").median()
            b = pd.to_numeric(df.loc[mask_ben, c], errors="coerce").median()
            note = ""
            if a == 0:
                note = "  <-- 0 = นับเฉพาะ payload ไม่รวม header"
            print(f"{c:34s} {a:12.1f} {b:12.1f}   {note}")
        print()
        print("  Suricata นับ bytes_toserver/toclient แบบ 'รวม header' (SYN = 60 bytes)")
        print("  ถ้าฝั่งนี้เป็น 0 แปลว่า CICFlowMeter นับเฉพาะ payload = คนละความหมาย")
        print("  -> feature ที่คำนวณจาก byte จะเทียบกันไม่ได้ ต้องตัดออก")
    print()


if __name__ == "__main__":
    main()
