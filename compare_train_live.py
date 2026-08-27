#!/usr/bin/env python3
"""
compare_train_live.py — เทียบ "ค่า feature ตอนเทรน" กับ "ค่า feature ตอน live"
ของการโจมตีชนิดเดียวกัน เพื่อหาว่าตัวไหนเพี้ยน

นี่คือเครื่องมือที่ควรมีตั้งแต่แรก: บั๊กทุกตัวที่เจอในโปรเจกต์นี้
(tcp flag ตาย, duration หยาบระดับวินาที) ล้วนเป็นเรื่องเดียวกันหมด คือ
"ค่าที่โมเดลเห็นตอนเทรน" กับ "ค่าที่โมเดลเห็นตอนใช้งานจริง" ไม่ตรงกัน
ทั้งที่ชื่อ feature เดียวกันเป๊ะ — และมันพังเงียบ ๆ ไม่มี error ให้เห็น

วิธีใช้: เอา flow ของการโจมตีจริง (เช่นยิง nmap -sS จาก Kali) มาเทียบกับ
row ที่ label ว่า PortScan ในชุดเทรน ถ้า feature ไหนค่าต่างกันเป็นสิบเท่า
ร้อยเท่า = เจอตัวการแล้ว

ใช้งาน:
  python3 compare_train_live.py \\
      --csv Dataset/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \\
      --label PortScan \\
      --eve /var/log/suricata/eve.json --src-ip 192.168.1.54

  # เทียบ DoS
  python3 compare_train_live.py --csv .../Wednesday-workingHours.pcap_ISCX.csv \\
      --label DoS --eve ... --src-ip ...
"""
import argparse
import json
import statistics

import pandas as pd

from hybrid_nids import HybridNIDS, FEATURES, load_and_preprocess


def _stats(vals):
    vals = [v for v in vals if v == v]  # ตัด NaN
    if not vals:
        return None
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ไฟล์ CICIDS ที่มี attack ชนิดนี้")
    ap.add_argument("--label", required=True,
                    help="ชื่อคลาสหลัง map เช่น PortScan / DoS / BruteForce")
    ap.add_argument("--eve", required=True, help="eve.json ที่มี traffic โจมตีจริง")
    ap.add_argument("--src-ip", dest="src_ip", required=True,
                    help="IP ผู้โจมตี (เช่น Kali) เพื่อกรองเอาเฉพาะ flow โจมตี")
    ap.add_argument("--dest-ip", dest="dest_ip", help="IP เป้าหมาย (ถ้าต้องการกรองเพิ่ม)")
    ap.add_argument("--max-live", type=int, default=5000)
    args = ap.parse_args()

    # ── ฝั่งเทรน ──────────────────────────────────────────────────
    print(f"อ่านชุดเทรน: {args.csv}")
    df = load_and_preprocess(args.csv)
    train = df[df["Label"] == args.label]
    if train.empty:
        print(f"ไม่พบ label '{args.label}' ในไฟล์นี้ "
              f"— ที่มี: {sorted(df['Label'].unique())}")
        return
    print(f"   rows ที่ label = {args.label}: {len(train):,}")

    # ── ฝั่ง live ─────────────────────────────────────────────────
    print(f"อ่าน live: {args.eve} (src_ip={args.src_ip})")
    nids = object.__new__(HybridNIDS)
    live_rows = []
    with open(args.eve, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(live_rows) >= args.max_live:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("event_type") != "flow":
                continue
            if raw.get("src_ip") != args.src_ip:
                continue
            if args.dest_ip and raw.get("dest_ip") != args.dest_ip:
                continue
            live_rows.append(nids._build_feature_row(raw))
    if not live_rows:
        print("   ไม่พบ flow จาก src_ip นี้เลย — ยิงโจมตีแล้วเก็บ eve.json ใหม่ก่อน")
        return
    live = pd.DataFrame(live_rows)
    print(f"   flow จาก {args.src_ip}: {len(live):,}\n")

    # ── เทียบทีละ feature ─────────────────────────────────────────
    print("=" * 96)
    print(f"เทียบ feature: ชุดเทรน({args.label})  vs  live({args.src_ip})")
    print("=" * 96)
    print(f"{'feature':24s} {'train median':>15s} {'live median':>15s} "
          f"{'อัตราส่วน':>12s}  สถานะ")
    print("-" * 96)

    suspects = []
    for col in FEATURES:
        t = _stats(pd.to_numeric(train[col], errors="coerce").tolist()) \
            if col in train.columns else None
        l = _stats(pd.to_numeric(live[col], errors="coerce").tolist()) \
            if col in live.columns else None
        if t is None or l is None:
            print(f"{col:24s} {'-':>15s} {'-':>15s} {'-':>12s}  ⚠️  ไม่มีข้อมูล")
            continue

        tm, lm = t["median"], l["median"]
        # ตัดสินว่า "ต่างกันมาก" ยังไง
        if tm == 0 and lm == 0:
            ratio, status = "1.00x", "✅"
        elif tm == 0 or lm == 0:
            ratio = "0 vs ไม่ 0"
            status = "❌ ตัวหนึ่งเป็น 0 อีกตัวไม่ใช่"
            suspects.append((col, tm, lm, "ฝั่งหนึ่งเป็น 0"))
        else:
            r = lm / tm
            ratio = f"{r:.2f}x"
            if r > 10 or r < 0.1:
                status = "❌ ต่างเกิน 10 เท่า"
                suspects.append((col, tm, lm, f"{r:.1f} เท่า"))
            elif r > 3 or r < 0.33:
                status = "⚠️  ต่าง 3-10 เท่า"
                suspects.append((col, tm, lm, f"{r:.1f} เท่า"))
            else:
                status = "✅"
        print(f"{col:24s} {tm:15.4g} {lm:15.4g} {ratio:>12s}  {status}")

    print("-" * 96)
    if suspects:
        print(f"\n❌ feature ที่ค่าต่างกันผิดปกติ {len(suspects)} ตัว "
              f"— นี่คือตัวที่ทำให้โมเดลทายเพี้ยน:\n")
        for col, tm, lm, why in suspects:
            print(f"     {col:24s} เทรน={tm:<14.4g} live={lm:<14.4g} ({why})")
        print("\n   feature พวกนี้ทำให้โมเดลเห็น 'การโจมตีแบบเดียวกัน' เป็นคนละอย่าง")
        print("   ระหว่างตอนเทรนกับตอนใช้จริง = train/serve skew")
        print("   ทางแก้: หาว่าฝั่งไหนคำนวณผิด แล้วแก้ให้ตรงกัน หรือถ้าแก้ไม่ได้")
        print("   (เช่น Suricata ไม่มีข้อมูลนั้นจริง ๆ) ให้ตัด feature นั้นออกจาก FEATURES")
    else:
        print("\n✅ ไม่พบ feature ที่ค่าต่างกันผิดปกติ")
        print("   ถ้าโมเดลยังทายผิดอยู่ แปลว่าปัญหาไม่ได้อยู่ที่ feature skew แล้ว")
        print("   แต่อยู่ที่ตัวโมเดล/threshold/กฎตัดสิน (เช่น SECONDARY_REJECTION_GAP)")
    print()


if __name__ == "__main__":
    main()
