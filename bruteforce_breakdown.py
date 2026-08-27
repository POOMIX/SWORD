"""
bruteforce_breakdown.py — เช็คว่า BruteForce task (F1=0.9977 ที่เพิ่งเทรน
ได้) แม่นเพราะเรียนรู้ syntax พฤติกรรมจริง หรือแค่ "เดา dest_port"
(FTP=21 / SSH=22 มี signature ชัดมาก ต่างจาก web-login brute force ที่
วิ่งบน port 80/443 ปนกับ traffic ปกติ ยากกว่ามาก) — ถ้า BruteForce ที่
เทรนมาแทบทั้งหมดมาจาก FTP/SSH (port 21/22) ตัวเลข F1 รวมที่เห็นอาจไม่ได้
สะท้อนว่าจับ web-login brute force (ที่เป็น 1 ใน 3 web attack ที่โปรเจกต์
นี้สนใจจริง ๆ) ได้ดีแค่ไหน

พิมพ์แยกทีละไฟล์ต้นทาง: จำนวนแถว BruteForce ต่อไฟล์ + distribution ของ
dest_port ในแถว BruteForce ของไฟล์นั้น

Usage:
    python bruteforce_breakdown.py --dataset-dir Dataset
"""

import argparse
import os

from hybrid_nids import load_and_preprocess

_C17 = "CICIDS2017/MachineLearningCVE"
_C18 = "CICIDS2018/CSV"

FILES = [
    (f"{_C17}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "web-login BF (2017)"),
    (f"{_C18}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", "web-login BF (2018)"),
    (f"{_C18}/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv", "web-login BF (2018)"),
    (f"{_C17}/Tuesday-WorkingHours.pcap_ISCX.csv", "network FTP/SSH-Patator (2017)"),
    (f"{_C18}/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", "network FTP/SSH-BruteForce (2018)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="Dataset")
    args = p.parse_args()

    total_bf = 0
    web_bf_rows = 0
    net_bf_rows = 0

    for fname, kind in FILES:
        path = os.path.join(args.dataset_dir, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} ไม่พบไฟล์")
            continue
        df = load_and_preprocess(path)
        bf = df[df["Label"] == "BruteForce"]
        n = len(bf)
        total_bf += n
        if "network" in kind:
            net_bf_rows += n
        else:
            web_bf_rows += n
        print(f"\n=== {fname}  ({kind}) ===")
        print(f"BruteForce rows: {n:,}")
        if n:
            print("dest_port distribution (top 10):")
            print(bf["dest_port"].value_counts().head(10).to_string())

    print(f"\n{'='*60}")
    print(f"สรุป: BruteForce รวม {total_bf:,} rows")
    print(f"  - จาก network FTP/SSH (port ชัดเจน, ง่าย)      : {net_bf_rows:,} "
          f"({net_bf_rows/max(total_bf,1)*100:.1f}%)")
    print(f"  - จาก web-login brute force (ปนกับ HTTP traffic, ยากกว่า) : {web_bf_rows:,} "
          f"({web_bf_rows/max(total_bf,1)*100:.1f}%)")
    if net_bf_rows / max(total_bf, 1) > 0.8:
        print("\n⚠️  ส่วนใหญ่ของ BruteForce ที่เทรนมาจาก FTP/SSH (port 21/22) ที่แยกง่าย "
              "F1=0.9977 ที่ได้อาจไม่ได้สะท้อนว่าจับ web-login brute force ได้ดีขนาดนั้น "
              "— ควรดู classification_report แยกเฉพาะ web-login subset ต่างหาก "
              "(บอกได้ถ้าอยากให้ผมเขียนสคริปต์แยก report เฉพาะ subtype)")


if __name__ == "__main__":
    main()
