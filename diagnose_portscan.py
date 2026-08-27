#!/usr/bin/env python3
"""
diagnose_portscan.py — หาสาเหตุว่าทำไม PortScan ถึงจับไม่ได้ แถมถูกทายเป็น
BruteForce

ตอบ 3 คำถามชี้ขาด (แต่ละส่วนรันแยกกันได้ ไม่มี arg ไหนก็ข้ามส่วนนั้นไป):

  [A] --eve       : Suricata flow event มี TCP flag/state ให้ใช้จริงไหม?
                    (ถ้ามี = เรากู้ feature ที่เป็นลายเซ็นของ PortScan
                     กลับมาได้แบบถูกต้อง ไม่ใช่ค่าปลอม)
  [B] --model_dir : โมเดลที่เทรนไว้ "จริง ๆ แล้วดูอะไร" ต่อ task
                    (feature importance ของ XGBoost) — ถ้า dest_port
                    ครองอันดับ 1 ขาดลอย แปลว่าโมเดลตัดสินจากเลข port
                    เป็นหลัก ซึ่งอธิบายได้ทันทีว่าทำไม flow ไป port 22
                    ถึงถูกทายเป็น BruteForce ไม่ว่าจะเป็น scan หรือไม่
  [C] --dataset   : ในข้อมูลเทรน PortScan กับ BruteForce ต่างกันตรงไหน
                    โดยเฉพาะการกระจายของ dest_port และ flag counts
                    (ยืนยันว่า label สองตัวนี้ "ทับกัน" ในพื้นที่ feature
                     ที่เหลืออยู่หรือเปล่า)

ตัวอย่าง:
  python3 diagnose_portscan.py --eve /var/log/suricata/eve.json \\
      --model_dir model --dataset /home/AiFromGit/SWORD_web/SWORD/Dataset
"""
import argparse
import json
import os
import pickle
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════
#  [A] Suricata flow event มี TCP flag/state ให้ใช้ไหม
# ══════════════════════════════════════════════════════════════════
def section_a(eve_path, limit_scan=200000):
    print("=" * 70)
    print("[A] ตรวจว่า Suricata flow event มี TCP flag / state ให้ใช้จริงไหม")
    print("=" * 70)

    n_flow = 0
    n_tcp_proto = 0
    n_has_tcp_obj = 0
    tcp_key_counter = Counter()
    flow_key_counter = Counter()
    samples = []
    state_counter = Counter()
    flowstate_counter = Counter()

    with open(eve_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= limit_scan:
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
            n_flow += 1
            if raw.get("proto") == "TCP":
                n_tcp_proto += 1
            tcp = raw.get("tcp")
            flow = raw.get("flow") or {}
            for k in flow.keys():
                flow_key_counter[k] += 1
            if isinstance(tcp, dict) and tcp:
                n_has_tcp_obj += 1
                for k in tcp.keys():
                    tcp_key_counter[k] += 1
                if tcp.get("state"):
                    state_counter[tcp["state"]] += 1
                if len(samples) < 5 and raw.get("proto") == "TCP":
                    samples.append({
                        "dest_port": raw.get("dest_port"),
                        "tcp": tcp,
                        "flow_state": flow.get("state"),
                        "flow_reason": flow.get("reason"),
                        "pkts": (flow.get("pkts_toserver"), flow.get("pkts_toclient")),
                    })
            if flow.get("state"):
                flowstate_counter[flow["state"]] += 1

    print(f"flow events ที่อ่าน       : {n_flow}")
    print(f"  ในนั้น proto == TCP     : {n_tcp_proto}")
    print(f"  ที่มี 'tcp' object ไม่ว่าง: {n_has_tcp_obj}")
    if n_tcp_proto:
        pct = 100.0 * n_has_tcp_obj / n_tcp_proto
        print(f"  => TCP flow ที่มี tcp obj: {pct:.1f}%")
    print()

    if n_has_tcp_obj:
        print(">>> มี TCP metadata ให้ใช้จริง! key ที่พบใน raw['tcp'] :")
        for k, c in tcp_key_counter.most_common():
            print(f"      {k:16s} พบ {c} ครั้ง")
        print()
        if state_counter:
            print("    การกระจายของ tcp.state :")
            for k, c in state_counter.most_common(10):
                print(f"      {k:16s} {c}")
            print()
        print("    ตัวอย่าง TCP flow event จริง :")
        for s in samples:
            print(f"      {json.dumps(s, ensure_ascii=False)}")
        print()
        print("    ✅ สรุป [A]: กู้ feature ลายเซ็น PortScan กลับมาได้แบบถูกต้อง")
        print("       (คำนวณสดจาก field พวกนี้จริง ไม่ใช่ค่าคงที่ปลอม)")
    else:
        print(">>> ไม่พบ 'tcp' object ใน flow event เลย")
        print("    ❌ สรุป [A]: ต้องแก้ด้วยวิธีอื่น (temporal fan-out feature)")
        print("       หรือเปิด metadata เพิ่มใน suricata.yaml ก่อน")

    print()
    print("    key ที่พบใน raw['flow'] (ใช้ดูว่ามีอะไรให้ใช้เพิ่มบ้าง) :")
    for k, c in flow_key_counter.most_common():
        print(f"      {k:20s} พบ {c} ครั้ง")
    if flowstate_counter:
        print("\n    การกระจายของ flow.state :")
        for k, c in flowstate_counter.most_common(10):
            print(f"      {k:16s} {c}")
    print()


# ══════════════════════════════════════════════════════════════════
#  [B] โมเดลจริง ๆ แล้วดู feature อะไร
# ══════════════════════════════════════════════════════════════════
def section_b(model_dir, top_n=15):
    print("=" * 70)
    print("[B] XGBoost feature importance ต่อ task — โมเดลตัดสินจากอะไร")
    print("=" * 70)

    feat_path = os.path.join(model_dir, "features.json")
    if not os.path.exists(feat_path):
        print(f"ไม่พบ {feat_path} — ข้ามส่วนนี้")
        return
    features = json.load(open(feat_path))
    print(f"จำนวน feature ที่โมเดลใช้: {len(features)}\n")

    for task in ["portscan", "dos", "bruteforce"]:
        p = os.path.join(model_dir, task, "xgb_model.pkl")
        if not os.path.exists(p):
            print(f"[{task}] ไม่พบ {p} — ข้าม\n")
            continue
        xgb = pickle.load(open(p, "rb"))
        try:
            imp = xgb.feature_importances_
        except AttributeError:
            print(f"[{task}] โมเดลไม่มี feature_importances_ — ข้าม\n")
            continue
        pairs = sorted(zip(features, imp), key=lambda kv: kv[1], reverse=True)
        total = float(sum(imp)) or 1.0
        print(f"[{task}] top {top_n} feature :")
        for name, v in pairs[:top_n]:
            share = 100.0 * float(v) / total
            bar = "█" * max(int(share / 2), 0)
            print(f"    {name:26s} {share:6.2f}%  {bar}")
        top_name, top_v = pairs[0]
        print(f"    -> อันดับ 1 คือ '{top_name}' ครองสัดส่วน "
              f"{100.0 * float(top_v) / total:.1f}% ของ importance ทั้งหมด")
        print()


# ══════════════════════════════════════════════════════════════════
#  [C] ในข้อมูลเทรน PortScan กับ BruteForce ต่างกันตรงไหน
# ══════════════════════════════════════════════════════════════════
_FILES = {
    "PortScan": ["CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
    "BruteForce": [
        "CICIDS2018/CSV/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "CICIDS2018/CSV/02-14-2018.csv",
        "CICIDS2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
    ],
}

_LABELS = {
    "PortScan": {"PortScan", "Port Scan"},
    "BruteForce": {"FTP-Patator", "SSH-Patator", "FTP-BruteForce",
                   "SSH-Bruteforce", "Web Attack \x96 Brute Force",
                   "Brute Force -Web"},
}


def section_c(dataset_dir):
    print("=" * 70)
    print("[C] ข้อมูลเทรน: PortScan vs BruteForce ต่างกันตรงไหน")
    print("=" * 70)
    try:
        import pandas as pd
    except ImportError:
        print("ไม่มี pandas — ข้ามส่วนนี้")
        return

    def _norm(c):
        return c.strip()

    for cls, rels in _FILES.items():
        found = False
        for rel in rels:
            path = os.path.join(dataset_dir, rel)
            if not os.path.exists(path):
                continue
            found = True
            try:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            except Exception as e:
                print(f"อ่าน {rel} ไม่ได้: {e}")
                continue
            df.columns = [_norm(c) for c in df.columns]
            label_col = next((c for c in df.columns if c.lower() == "label"), None)
            if not label_col:
                print(f"{rel}: ไม่พบคอลัมน์ Label — ข้าม")
                continue
            labels = df[label_col].astype(str).str.strip()
            mask = labels.isin(_LABELS[cls])
            sub = df[mask]
            print(f"\n--- {cls}  ({rel}) : {len(sub)} rows ---")
            if sub.empty:
                print(f"    label ที่มีในไฟล์นี้: {sorted(labels.unique())[:12]}")
                continue

            dport_col = next((c for c in sub.columns
                              if c.lower().replace(" ", "") in
                              ("destinationport", "dstport", "destport")), None)
            if dport_col:
                vc = sub[dport_col].value_counts()
                print(f"    dest_port ที่พบบ่อยสุด 10 อันดับ (จาก {sub[dport_col].nunique()} ค่าไม่ซ้ำ):")
                for port, cnt in vc.head(10).items():
                    print(f"        port {str(port):>7s} : {cnt:>8d} rows "
                          f"({100.0 * cnt / len(sub):5.1f}%)")
                print(f"        -> จำนวน port ไม่ซ้ำ = {sub[dport_col].nunique()}")

            for key in ("SYN Flag Count", "RST Flag Count", "ACK Flag Count",
                        "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt",
                        "Flow Duration", "Total Fwd Packets", "Tot Fwd Pkts"):
                col = next((c for c in sub.columns if c.lower() == key.lower()), None)
                if col is not None:
                    s = pd.to_numeric(sub[col], errors="coerce")
                    print(f"    {key:20s} mean={s.mean():12.3f}  median={s.median():10.2f}")
            break
        if not found:
            print(f"\n--- {cls}: ไม่พบไฟล์ที่คาดไว้ใต้ {dataset_dir} ---")
            for rel in rels:
                print(f"      ลองหา: {rel}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eve", help="path ไปยัง eve.json (ส่วน A)")
    ap.add_argument("--model_dir", help="โฟลเดอร์โมเดลที่เทรนแล้ว (ส่วน B)")
    ap.add_argument("--dataset", help="โฟลเดอร์ Dataset (ส่วน C)")
    ap.add_argument("--scan-limit", type=int, default=200000,
                    help="อ่าน eve.json กี่บรรทัด (default 200000)")
    args = ap.parse_args()

    if not any([args.eve, args.model_dir, args.dataset]):
        ap.error("ต้องระบุอย่างน้อยหนึ่งใน --eve / --model_dir / --dataset")

    if args.eve:
        section_a(args.eve, args.scan_limit)
    if args.model_dir:
        section_b(args.model_dir)
    if args.dataset:
        section_c(args.dataset)


if __name__ == "__main__":
    main()
