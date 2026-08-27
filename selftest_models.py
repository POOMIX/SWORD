#!/usr/bin/env python3
"""
selftest_models.py — ตรวจความพร้อมของโมเดลทั้งหมด "บนเครื่องที่จะรันจริง"
ก่อนเริ่ม --realtime

ทำไมต้องมี: โปรเจกต์นี้เทรนบนเครื่องหนึ่ง (server ที่แรงกว่า) แล้วเอา
ไฟล์ .pkl มารันอีกเครื่องหนึ่ง ซึ่งเป็นเรื่องปกติและทำได้ แต่มีความเสี่ยง
ที่มองไม่เห็นอยู่ 2 อย่าง:

  1) model_dir ไม่ครบ — flow model กับ payload model ต้องอยู่โฟลเดอร์
     เดียวกันบนเครื่องที่รัน ถ้าเทรนคนละเครื่องแล้วย้ายมาไม่ครบ ระบบจะ
     ทำงานต่อแบบเงียบ ๆ (signature-only) โดยไม่มี error

  2) เวอร์ชันไลบรารีไม่ตรงกัน — pickle ของ scikit-learn/XGBoost ผูกกับ
     เวอร์ชันที่ตอนสร้าง ถ้าเครื่องที่รันมีเวอร์ชันต่างจากเครื่องที่เทรน
     อาจ (ก) โหลดไม่ขึ้นเลย หรือแย่กว่านั้นคือ (ข) โหลดขึ้นแต่ให้ผลเพี้ยน
     แบบไม่มีคำเตือน — ซึ่งเป็นบั๊กประเภทเดียวกับที่ทำให้โปรเจกต์นี้เสีย
     เวลามาเยอะแล้ว (พังเงียบ ๆ ไม่มี error)

สคริปต์นี้จึงโหลดโมเดลจริงทุกตัว เทียบเวอร์ชัน แล้ว "ยิงข้อมูลตัวอย่าง
เข้าไปจริง" เพื่อดูว่าให้คำตอบสมเหตุสมผลไหม — ไม่ใช่แค่เช็คว่าไฟล์มีอยู่

ใช้งาน:
    python3 selftest_models.py --model_dir model
"""
import argparse
import json
import os
import pickle
import sys

import pandas as pd

OK = "✅"
BAD = "❌"
WARN = "⚠️ "


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./model")
    args = ap.parse_args()
    md = os.path.abspath(args.model_dir)
    problems = []
    warnings_ = []

    # ── 1. เวอร์ชันไลบรารีบนเครื่องนี้ ────────────────────────────
    section("1. เวอร์ชันไลบรารีบนเครื่องที่กำลังรัน")
    import sklearn
    import numpy
    import xgboost
    vers = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }
    for k, v in vers.items():
        print(f"    {k:14s} {v}")

    # ── 2. ไฟล์ครบไหม ─────────────────────────────────────────────
    section(f"2. ไฟล์ในโมเดล: {md}")
    if not os.path.isdir(md):
        print(f"{BAD} ไม่พบโฟลเดอร์ {md}")
        sys.exit(1)

    flow_tasks = ["portscan", "dos", "bruteforce"]
    per_task = ["dt_model.pkl", "rf_model.pkl", "xgb_model.pkl", "scaler.pkl"]
    root_files = ["features.json", "thresholds.json"]
    payload_files = ["payload_vectorizer.pkl", "payload_xgb.pkl", "payload_meta.json"]

    for fn in root_files:
        p = os.path.join(md, fn)
        print(f"    {OK if os.path.exists(p) else BAD} {fn}")
        if not os.path.exists(p):
            problems.append(f"ขาด {fn}")

    p = os.path.join(md, "thresholds_xgb.json")
    if os.path.exists(p):
        print(f"    {OK} thresholds_xgb.json (โหมด --xgb_only ใช้ได้)")
    else:
        print(f"    {WARN}thresholds_xgb.json ไม่มี — --xgb_only จะยืม "
              f"threshold ของ ensemble ซึ่งคนละสเกล")
        warnings_.append("ไม่มี thresholds_xgb.json (เทรนด้วยโค้ดเก่ากว่า v4.0)")

    for t in flow_tasks:
        miss = [f for f in per_task if not os.path.exists(os.path.join(md, t, f))]
        if miss:
            print(f"    {BAD} {t}/ ขาด: {miss}")
            problems.append(f"{t} ขาดไฟล์ {miss}")
        else:
            print(f"    {OK} {t}/ ครบ 4 ไฟล์")

    print()
    pay_miss = [f for f in payload_files if not os.path.exists(os.path.join(md, f))]
    if pay_miss:
        print(f"    {BAD} Payload ML ขาด: {pay_miss}")
        print(f"        -> ระบบจะรันแบบ signature-only (จับ XSS/SQLi ได้น้อยลงมาก)")
        print(f"        -> แก้: python3 train_payload.py web_payloads_merged.csv --model_dir {md}")
        problems.append(f"Payload ML ขาดไฟล์ {pay_miss}")
    else:
        print(f"    {OK} Payload ML ครบ 3 ไฟล์ (อยู่ model_dir เดียวกับ flow model)")

    # ── 3. features.json ตรงกับโค้ดไหม ────────────────────────────
    section("3. feature set ของโมเดล vs โค้ดบนเครื่องนี้")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hybrid_nids import FEATURES, USE_TCP_FEATURES
        saved = json.load(open(os.path.join(md, "features.json")))
        print(f"    โมเดลเทรนด้วย : {len(saved)} features")
        print(f"    โค้ดสร้างได้   : {len(FEATURES)} features  "
              f"(USE_TCP_FEATURES={USE_TCP_FEATURES})")
        if list(saved) == list(FEATURES):
            print(f"    {OK} ตรงกันเป๊ะ")
        else:
            only_m = [f for f in saved if f not in FEATURES]
            only_c = [f for f in FEATURES if f not in saved]
            print(f"    {BAD} ไม่ตรงกัน!")
            print(f"        มีเฉพาะในโมเดล: {only_m or '-'}")
            print(f"        มีเฉพาะในโค้ด : {only_c or '-'}")
            print(f"        -> ต้อง retrain ด้วยโค้ดเวอร์ชันนี้ หรือปรับ "
                  f"USE_TCP_FEATURES ให้ตรงกับตอนเทรน")
            problems.append("feature set ไม่ตรงกับโค้ด")
    except Exception as e:
        print(f"    {BAD} ตรวจไม่ได้: {type(e).__name__}: {e}")
        problems.append(f"ตรวจ features ไม่ได้: {e}")

    # ── 4. เวอร์ชันที่ฝังใน pickle ตรงกับเครื่องนี้ไหม ──────────────
    section("4. เวอร์ชันตอนเทรน (ฝังอยู่ใน .pkl) vs เครื่องนี้")
    print("    scikit-learn ฝังเวอร์ชันไว้ใน estimator ตอน pickle — ถ้าไม่ตรง")
    print("    กับเครื่องที่รัน อาจให้ผลเพี้ยนแบบไม่มี error เตือน\n")
    seen = set()
    for t in flow_tasks:
        for fn in ("rf_model.pkl", "xgb_model.pkl", "scaler.pkl"):
            p = os.path.join(md, t, fn)
            if not os.path.exists(p):
                continue
            try:
                obj = pickle.load(open(p, "rb"))
            except Exception as e:
                print(f"    {BAD} {t}/{fn} โหลดไม่ขึ้น: {type(e).__name__}: {e}")
                problems.append(f"{t}/{fn} โหลดไม่ขึ้น")
                continue
            v = getattr(obj, "_sklearn_version", None)
            if v:
                seen.add(v)
    if not seen:
        print(f"    {WARN}อ่านเวอร์ชันจาก pickle ไม่ได้ (ไม่ใช่ปัญหาเสมอไป)")
    else:
        for v in sorted(seen):
            if v == vers["scikit-learn"]:
                print(f"    {OK} เทรนด้วย scikit-learn {v} = ตรงกับเครื่องนี้")
            else:
                print(f"    {BAD} เทรนด้วย scikit-learn {v} "
                      f"แต่เครื่องนี้เป็น {vers['scikit-learn']}")
                print(f"        -> ให้ตรงกันด้วย: pip install scikit-learn=={v}")
                problems.append(f"scikit-learn ไม่ตรง ({v} vs {vers['scikit-learn']})")

    # ── 5. ยิงข้อมูลจริงเข้าไปดูว่าตอบสมเหตุสมผลไหม ────────────────
    section("5. Smoke test — ยิงตัวอย่างจริงเข้าโมเดล")
    try:
        from hybrid_nids import HybridNIDS
        nids = HybridNIDS(model_dir=md)
        print(f"    โหลดโมเดลสำเร็จ — tasks: {list(nids.models_ovr.keys())}")
        print(f"    thresholds: {nids.tuned_thresholds}\n")

        cases = {
            "nmap -sS ยิง port ปิด (ควรเป็น PortScan/Benign ไม่ใช่ BruteForce)": {
                "proto": "TCP", "dest_port": 22,
                "flow": {"pkts_toserver": 1, "pkts_toclient": 1,
                          "bytes_toserver": 60, "bytes_toclient": 60, "age": 0.001},
                "tcp": {"syn": True, "rst": True, "ack": True,
                         "fin": False, "psh": False, "urg": False}},
            "ssh login attempt (ควรเอียงไป BruteForce)": {
                "proto": "TCP", "dest_port": 22,
                "flow": {"pkts_toserver": 14, "pkts_toclient": 12,
                          "bytes_toserver": 1800, "bytes_toclient": 2400, "age": 0.4},
                "tcp": {"syn": True, "rst": False, "ack": True,
                         "fin": True, "psh": True, "urg": False}},
            "เปิดเว็บปกติ (ควรเป็น Benign)": {
                "proto": "TCP", "dest_port": 443,
                "flow": {"pkts_toserver": 22, "pkts_toclient": 30,
                          "bytes_toserver": 5000, "bytes_toclient": 40000, "age": 4.0},
                "tcp": {"syn": True, "rst": False, "ack": True,
                         "fin": True, "psh": True, "urg": False}},
        }
        for name, ev in cases.items():
            row = nids._build_feature_row(ev)
            res = nids.predict(pd.DataFrame([row]))
            scores = ", ".join(f"{k}={v:.3f}" for k, v in
                               sorted(res["all_scores"].items(),
                                      key=lambda kv: -kv[1]))
            print(f"    {name}")
            print(f"        -> {res['predicted_class']}  ({scores})")
        print()

        if nids.payload_xgb is not None:
            for s, want in [("/?id=1' OR '1'='1", "SQL Injection"),
                            ("/?q=<script>alert(1)</script>", "XSS"),
                            ("/index.php?page=home", "ไม่ควรแจ้ง")]:
                spec, conf = nids._run_payload_ml(s)
                print(f"    payload {s!r}\n        -> {spec or 'Benign'} "
                      f"(conf={conf:.3f}) | คาดหวัง: {want}")
        else:
            print(f"    {WARN}ข้าม smoke test ของ payload ML (ไม่ได้โหลด)")
    except SystemExit as e:
        print(f"    {BAD} โหลดโมเดลไม่ได้ (ดูข้อความด้านบน)")
        problems.append("โหลดโมเดลไม่ได้")
    except Exception as e:
        print(f"    {BAD} smoke test ล้มเหลว: {type(e).__name__}: {e}")
        problems.append(f"smoke test ล้มเหลว: {e}")

    # ── สรุป ──────────────────────────────────────────────────────
    section("สรุป")
    for w in warnings_:
        print(f"    {WARN}{w}")
    if problems:
        print(f"    {BAD} พบปัญหา {len(problems)} เรื่อง — ควรแก้ก่อนรันจริง:")
        for p_ in problems:
            print(f"        - {p_}")
        sys.exit(1)
    print(f"    {OK} ทุกอย่างพร้อม — รัน --realtime ได้เลย")
    sys.exit(0)


if __name__ == "__main__":
    main()
