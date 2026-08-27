"""
debug_flow_scores.py — ดึง flow event จริงจาก eve.json (เฉพาะ dest_ip ที่ระบุ)
ป้อนเข้าโมเดลที่โหลดจาก model_dir ตรงๆ แล้วพิมพ์คะแนนดิบ (raw ensemble score)
ของทุก task (portscan/dos/bruteforce) เทียบกับ tuned threshold + secondary
rejection gap (0.15) ออกมาให้เห็นชัดๆ ว่าทำไม predict() ถึงไม่ฟันธงว่าเป็น
attack — ไม่ต้องรอ --realtime ทั้งระบบ เอาไฟล์ eve.json ที่มีอยู่แล้วมาย้อน
วิเคราะห์ตรงๆ

ใช้งาน:
  python3 debug_flow_scores.py --eve /var/log/suricata/eve.json \\
      --dest-ip 192.168.1.52 --model_dir model [--limit 30]

จะพิมพ์: ทุก flow event ที่ dest_ip ตรง พร้อม dest_port, pkts/bytes,
คะแนนดิบของ portscan/dos/bruteforce, threshold ที่ tune ไว้, gap ระหว่าง
top-2, และสรุปว่า is_attack ตัดสินว่าอย่างไรและเพราะอะไร (score ไม่ถึง
threshold, หรือ gap ไม่พอ)
"""
import argparse
import json
import sys

import pandas as pd

# import HybridNIDS class ตรงจาก hybrid_nids.py ที่อยู่ไดเรกทอรีเดียวกัน
from hybrid_nids import HybridNIDS, SECONDARY_REJECTION_GAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eve", required=True, help="path ไปยัง eve.json")
    ap.add_argument("--dest-ip", required=True, help="dest_ip ที่จะกรอง (เช่น IP ของ target ที่โดนสแกน)")
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--limit", type=int, default=30, help="พิมพ์กี่ event แรกที่เจอ (default 30)")
    args = ap.parse_args()

    print(f"กำลังโหลดโมเดลจาก {args.model_dir} ...")
    nids = HybridNIDS(model_dir=args.model_dir)
    print(f"โหลดสำเร็จ — tasks ที่มี: {list(nids.models_ovr.keys())}")
    print(f"tuned_thresholds: {nids.tuned_thresholds}")
    print(f"SECONDARY_REJECTION_GAP: {SECONDARY_REJECTION_GAP}\n")

    shown = 0
    total_flow_events = 0
    with open(args.eve, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("event_type") != "flow":
                continue
            if raw.get("dest_ip") != args.dest_ip:
                continue
            total_flow_events += 1
            if shown >= args.limit:
                continue

            row = nids._build_feature_row(raw)
            res = nids.predict(pd.DataFrame([row]))

            scores = res.get("all_scores", {})
            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_score = sorted_scores[0] if sorted_scores else (None, 0.0)
            second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
            gap = top_score - second_score
            eff_t = nids.tuned_thresholds.get(top_label, nids.threshold)

            flow = raw.get("flow", {}) or {}
            print(f"--- flow_id={raw.get('flow_id')} src={raw.get('src_ip')}:{raw.get('src_port')} "
                  f"-> {raw.get('dest_ip')}:{raw.get('dest_port')} proto={raw.get('proto')} "
                  f"pkts(s/d)={flow.get('pkts_toserver')}/{flow.get('pkts_toclient')} "
                  f"bytes(s/d)={flow.get('bytes_toserver')}/{flow.get('bytes_toclient')} ---")
            for label, s in sorted_scores:
                t = nids.tuned_thresholds.get(label, nids.threshold)
                mark = " <== เกิน threshold" if s >= t else ""
                print(f"    {label:12s} score={s:.4f}  threshold={t:.4f}{mark}")
            print(f"    top={top_label} score={top_score:.4f} vs threshold={eff_t:.4f} "
                  f"| gap(top-2nd)={gap:.4f} (ต้อง >= {SECONDARY_REJECTION_GAP})")
            print(f"    => predict() ตัดสิน: predicted_class={res.get('predicted_class')} "
                  f"is_attack={res.get('is_attack')}")
            if not res.get("is_attack"):
                if top_score < eff_t:
                    print(f"    เหตุผล: score ({top_score:.4f}) ไม่ถึง threshold ({eff_t:.4f})")
                elif gap < SECONDARY_REJECTION_GAP:
                    print(f"    เหตุผล: score ถึง threshold แล้ว แต่ gap ({gap:.4f}) < "
                          f"{SECONDARY_REJECTION_GAP} — โดน secondary rejection")
            print()
            shown += 1

    print(f"\nสรุป: เจอ flow event ทั้งหมดที่ dest_ip={args.dest_ip} จำนวน {total_flow_events} "
          f"(พิมพ์รายละเอียด {shown} รายการแรก)")


if __name__ == "__main__":
    main()
