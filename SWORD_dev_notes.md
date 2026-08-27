# SWORD — Developer Notes / Handoff

> เอกสารนี้สรุป context การพัฒนา `hybrid_nids.py` ทั้งหมด สำหรับให้ Claude
> Code (หรือคนอื่น) อ่านแล้วสานงานต่อได้ทันทีโดยไม่ต้องเริ่มใหม่
> **อ่านไฟล์นี้ก่อนแก้ `hybrid_nids.py` เสมอ** — มีบั๊กหลายตัวที่ "ดูเหมือน
> ควรแก้แบบหนึ่ง" แต่จริง ๆ ต้องแก้อีกแบบ เขียนเหตุผลไว้ครบแล้ว
>
> เวอร์ชันล่าสุดของโค้ด: **v5.9** (ดู `VERSION` ต้นไฟล์ `hybrid_nids.py`)

---

## 1. SWORD คืออะไร (ภาพรวม)

Hybrid Network IDS สำหรับวิทยานิพนธ์ ป.ตรี ประกอบด้วย 3 ชั้นตรวจจับ:

1. **Flow-based ML** (OvR ensemble: DecisionTree + RandomForest + XGBoost)
   ต่อ task แบบ binary — เทรนจาก CICIDS2017/2018 (CICFlowMeter CSV)
2. **Payload-based ML** (TF-IDF + XGBoost multiclass) — จับ XSS / SQLi จาก
   เนื้อหา HTTP
3. **Temporal / statistical layers** — จับสิ่งที่ ML ต่อ flow เดียวมองไม่เห็น
   (PortScan fan-out, DoS rate anomaly, BruteForce repetition)

**สภาพแวดล้อม:**
- โค้ดอยู่ 2 ที่: `D:\Github\SWORD` (Windows, git repo) และ
  `~/SWORD_web/SWORD` หรือ `~/Desktop/Project/SWORD` (Linux, ที่เทรน+รันจริง)
- Linux = Ubuntu, Suricata 7.0.3, interface `enp0s8`
- เครื่อง SWORD = target (`192.168.1.52`), Kali = attacker (`192.168.1.54`)
- รันด้วย venv: `source venv/bin/activate` หรือ `venv/bin/python`
- **`sudo` ไม่ใช้ venv** — ต้อง `sudo venv/bin/python ...` ถ้าต้อง root
  (แต่ปกติรัน `--realtime` แบบไม่ sudo ได้ ถ้าอ่าน eve.json ได้)

---

## 2. สถาปัตยกรรมสุดท้าย (v5.9) — ใครจับด้วยอะไร

| การโจมตี | วิธีตรวจ (runtime) | ML หรือ rule | หมายเหตุ |
|---|---|---|---|
| **PortScan** | temporal fan-out (นับ distinct port ต่อ src) | rule + ML ยืนยัน | per-flow ML PortScan **ถูกตัด** (unreliable) |
| **DoS** | adaptive rate anomaly (z-score เทียบ baseline) | statistical | per-flow ML DoS **ถูกตัด** (v5.8) |
| **BruteForce** | per-flow ML + repetition gate (ซ้ำ ≥4 ใน 20s) | ML + rule | ยังใช้ ML จริง |
| **WebAttack (XSS/SQLi)** | payload ML (TF-IDF+XGBoost) | ML | flow-level webattack ML **ปิด** (precision ต่ำ) |
| **Suricata signature** | `event_type=="alert"` จาก Suricata | signature | เส้นทางแยก ไม่เกี่ยวกับ ML ภายใน |

**หลักการสำคัญ (เขียนลงเล่มได้):**
> การโจมตีที่เป็น "ปรากฏการณ์ข้ามหลาย flow" (PortScan = fan-out, DoS =
> rate) ไม่มี per-flow classifier ตัวไหนจับได้จาก flow เดียว เพราะสัญญาณ
> อยู่ "ระหว่าง flow" → ใช้ temporal/statistical layer / ส่วนที่ดูได้จาก
> flow หรือ payload เดียว (BruteForce, Web) ใช้ ML — นี่คือความหมายที่แท้จริง
> ของคำว่า Hybrid IDS

`_RUNTIME_FLOW_TASKS = ["portscan", "dos", "bruteforce"]` แต่ในโค้ด
`_analyze_flow` / `_analyze_http_event` จะ **suppress** ผลของ portscan และ
dos จาก per-flow ML (return ก่อนยิง alert) เพราะไม่น่าเชื่อถือ — เก็บโมเดล
ไว้เทียบผล offline ในเล่มได้ แต่ไม่ใช้ยิง alert สด

---

## 3. FEATURES (25 ตัว) — ระวังมากตรงนี้

```
dest_port, duration, total_fwd_packets, total_bwd_packets, total_packets,
flow_packets_per_sec, duration_ms, fwd_bwd_ratio, pkt_ratio, has_response,
flow_iat_mean, is_long_connection, log_duration, pkts_per_duration, acc_age,
n_flushes, log_acc_age, http_request_count, http_method_count,
http_status_4xx_ratio, http_status_5xx_ratio, http_uri_len_avg,
http_uri_len_max, http_param_count, has_suspicious_chars
```

**กฎเหล็ก: feature ทุกตัวต้องคำนวณได้ "เหมือนกันเป๊ะ" ทั้งตอนเทรน (จาก
CICIDS CSV) และตอน live (จาก Suricata eve.json)** ไม่งั้นเกิด train/serve
skew = offline metric สวย แต่ live พังเงียบ ๆ (Arp et al. 2022, P9)

มี compatibility check ใน `_load_models()` — ถ้า `features.json` ของโมเดล
ไม่ตรงกับ `FEATURES` ในโค้ด จะ **ปฏิเสธการรันทันที** ไม่ปล่อยให้ทายมั่ว

### feature ที่ "ตัดออกแล้ว" และห้ามเอากลับ (มีเหตุผล):
- **tcp_syn/rst/ack/fin/psh** — พิสูจน์ด้วย `check_cicids_flags.py` ว่าค่าใน
  CICIDS2017 ไม่สอดคล้องความจริง (PortScan มี SYN 0%, PSH 100% ซึ่งเป็นไป
  ไม่ได้ — เป็นบั๊กของ CICFlowMeter, อ้าง Engelen et al. 2021) → ตั้ง
  `USE_TCP_FEATURES = False`
- **byte-per-packet features** (fwd_bytes_per_pkt, bytes_ratio, ...) —
  CICFlowMeter นับ "เฉพาะ payload" แต่ Suricata นับ "รวม header" = คนละ
  นิยาม แปลงให้ตรงกันไม่ได้
- **down_up_ratio** — CICIDS เป็นอัตราส่วน packet แต่โค้ด live เคยคำนวณเป็น
  อัตราส่วน byte = คนละความหมาย

### เครื่องมือ diagnostic (อยู่ในโปรเจกต์):
- `hybrid_nids.py --verify-live <eve.json>` — ตรวจว่า feature ตัวไหนเป็นค่า
  คงที่ (dead) ตอน live = train/serve skew **รันก่อน --train เสมอ**
- `compare_train_live.py` — เทียบค่า feature ตอนเทรน vs ตอน live ทีละตัว
- `check_cicids_flags.py` — อ่านค่าดิบใน CICIDS CSV (ตรวจว่าเป็นบั๊ก dataset
  หรือบั๊กเรา)
- `debug_flow_scores.py` — ดูคะแนน ML + feature ที่โมเดลเห็น ต่อ flow
- `selftest_models.py` — ตรวจโมเดลบนเครื่องที่จะรันจริงก่อน --realtime

---

## 4. Temporal / statistical layers — ค่าที่ปรับได้ (ต้นไฟล์)

### PortScan (fan-out)
```python
PORTSCAN_MIN_DISTINCT_PORTS = 15    # แตะ >= N port ไม่ซ้ำ = scan
PORTSCAN_WINDOW_SECONDS = 60.0
PORTSCAN_SCANLIKE_MAX_BWD_BYTES = 100   # นับเฉพาะ flow ที่ปลายทางแทบไม่ตอบ
PORTSCAN_SCANLIKE_MAX_BWD_PKTS = 2
```

### BruteForce (repetition gate)
```python
BRUTEFORCE_MIN_ATTEMPTS = 4          # ML ทาย BruteForce ซ้ำ >= N ครั้ง
BRUTEFORCE_WINDOW_SECONDS = 20.0     # ภายใน N วินาที ถึงยิง alert
```

### DoS (adaptive rate anomaly) — ซับซ้อนสุด อ่านให้ดี
```python
DOS_RATE_WINDOW_SECONDS = 10.0       # หน้าต่างวัดอัตรา (ยาวพอเฉลี่ย burst ปกติทิ้ง)
DOS_ROBUST_Z = 6.0                   # flag เมื่ออัตราเกิน baseline นี้ (robust z)
DOS_MIN_RATE = 5.0                   # noise floor: ต่ำกว่านี้ (req/s) ไม่ flag
DOS_EWMA_ALPHA = 0.03                # ความเร็วปรับ baseline
DOS_PRIOR_RATE = 1.0                 # baseline เริ่มต้น (กัน cold-start poisoning)
DOS_BASELINE_UPDATE_INTERVAL = 5.0   # อัปเดต baseline ตาม "เวลา" ไม่ใช่ทุก event
DOS_FLOOD_COOLDOWN_SECONDS = 15.0    # หลังจับ flood ลัดวงจร (ข้าม ML) กัน XSS FP
DOS_BASELINE_FILE = "dos_baseline.json"   # persist baseline ข้ามรอบ
```

**หลักการ DoS detector (สำคัญมาก — เคยพังหลายรอบ):**
- ไม่ใช้เลข "จำนวนครั้ง" ตายตัว (hardcode) — ผู้ใช้ปฏิเสธเพราะ DoS ช้าจะหลุด
- เรียนรู้ baseline ของ "อัตราต่อต้นทาง" ที่แต่ละบริการ `(dst_ip,dst_port)`
  เห็นตามปกติ แล้ว flag ต้นทางที่อัตราเป็น outlier (z-score)
- **บั๊กที่แก้แล้ว 3 ตัว (ห้ามทำพัง):**
  1. **cold-start poisoning**: ถ้าบริการโดนโจมตีตั้งแต่แรกโดยไม่มี traffic
     ปกติมาก่อน ระบบจะเรียน baseline จากตัวการโจมตี → แก้ด้วย `DOS_PRIOR_RATE`
     ต่ำ + poisoning guard (อัปเดต baseline เฉพาะค่า <= mean+3·mad)
  2. **event-volume poisoning**: flood 700 req/s = อัปเดต baseline 700 ครั้ง/s
     → baseline ไล่ตามทัน → แก้ด้วย `DOS_BASELINE_UPDATE_INTERVAL` (อัปเดต
     ตามเวลา ไม่ใช่ทุก event)
  3. **normal burst FP**: page load ยิง 25 req/1s → แก้ด้วยหน้าต่างยาว 10s
     เฉลี่ย burst สั้นทิ้ง เหลือ sustained flood จริง

---

## 5. เวลา event vs เวลาประมวลผล — บั๊กสำคัญ (v5.6)

**temporal detector ทุกตัวต้องใช้ `_event_time(raw)` = เวลาที่ event เกิด
จริง (จาก `raw["timestamp"]`) ไม่ใช่ `time.time()` (เวลาประมวลผล)**

เพราะเวลาโดน flood → queue backlog → event ถูกประมวลผลช้ากว่าที่เกิดจริง
มาก ถ้าคิด "อัตรา" จากเวลาประมวลผล จะเพี้ยน (อัตราถูกยืด) → จับ DoS ช้า/พลาด
และ event ของ flood หลุดไปถึง payload ML จน false positive

`handle_alert` dedup ยังใช้ `time.time()` (wall clock) ได้ — เพราะเป็นการ
คุมอัตราการ "พิมพ์ alert" ไม่ใช่การตรวจจับ

---

## 6. ประวัติเวอร์ชัน (ทำไมแต่ละอันเกิดขึ้น)

| Ver | ทำอะไร | ต้อง retrain? |
|---|---|---|
| 3.2 | เวอร์ชันเก่าที่ "เคยใช้ได้" — 17 features, ไม่มี gap rule, ไม่มี tcp/byte | (baseline) |
| 4.0 | เอา TCP flag กลับมา (ผิด — ดูด้านล่าง) + temporal PortScan | ✅ |
| 4.1 | ตัด derived tcp ที่ตายบน live | ✅ |
| 4.2 | แก้ข้อความ payload-not-found ให้บอก path | ไม่ |
| 5.0 | duration ใช้ flow.start/end (ไมโครวินาที) แทน flow.age (วินาที) | ไม่ |
| 5.1 | **ตัด tcp + byte features → 25 ตัว** (พิสูจน์ว่า CICIDS เพี้ยน) + ลด gap 0.15→0.05 | ✅ |
| 5.2 | ปิด signature-override web (ML ล้วน) + payload per-source scoring | ไม่ |
| 5.3 | **shared benign pool** (แก้ DoS FP: DHCP/DNS) — ทุก task เทรน benign ชุดเดียว | ✅ |
| 5.4 | DoS flood gate (hardcode) + แก้ PortScan FP จาก sqlmap | ไม่ |
| 5.5 | เปลี่ยน DoS เป็น adaptive (ทิ้ง hardcode count) | ไม่ |
| 5.6 | event-time (แก้ backlog) + flood cooldown (แก้ XSS FP) | ไม่ |
| 5.7 | **แก้ cold-start poisoning** + sustained-rate window + persist baseline | ไม่ |
| 5.8 | ตัด per-flow ML DoS (แก้ PortScan probe → DoS FP) | ไม่ |
| 5.9 | warning duration นับเฉพาะ flow event จริง (ไม่หลอกตา) | ไม่ |

**สรุป: retrain ครั้งสุดท้ายที่จำเป็นคือ v5.3** — หลังจากนั้นทุกอันแก้ตรรกะ
inference ล้วน โมเดลปัจจุบัน (25 features, shared benign pool) ใช้ได้กับ
โค้ด v5.9

**การเทรน (ถ้าต้อง):**
```bash
python3 hybrid_nids.py --train <DATASET_DIR> --model_dir model
# DATASET_DIR มี CICIDS2017/MachineLearningCVE/ และ CICIDS2018/CSV/
# ใช้เวลา ~20-30 นาที บนเครื่อง 6-core
```
โหมดรันจริง (เร็ว): `--xgb_only` (ใช้ XGB เดี่ยว ข้าม DT/RF, threshold แยก
ใน `thresholds_xgb.json`) — คอขวดหลักคือ CalibratedClassifierCV(RF 300 ต้น)
ensemble เต็ม = 1.1s/flow, xgb_only = 0.018s/flow (เร็วกว่า 62 เท่า)

---

## 7. บทเรียนใหญ่ (train/serve skew ทั้งหมดที่เจอ)

ทุกครั้งที่ "offline F1 สวย แต่ live ไม่ทำงาน" = train/serve skew เสมอ:
1. **tcp flags** — CICIDS CSV มีค่า (แต่เพี้ยน), Suricata live ไม่มี/ต่างนิยาม
2. **duration** — CICIDS = ไมโครวินาที, flow.age = วินาที (แก้ด้วย start/end)
3. **byte features** — CICIDS = payload only, Suricata = รวม header
4. **PortScan features กลับด้าน** — flow scan ในแล็บ (closed port→RST) ต่างจาก
   CICIDS (filtered port→ไม่ตอบ) ทำให้ has_response/pkt_ratio กลับด้าน

→ วิธีกัน: `--verify-live` + `compare_train_live.py` ก่อน deploy เสมอ

---

## 8. ข้อจำกัดที่รู้ตัว (เขียนบท Limitation ในเล่มได้เลย)

1. **HTTPS**: payload ML (XSS/SQLi) ตรวจไม่ได้ เพราะ payload เข้ารหัส
   Suricata เห็นแค่ TLS metadata (SNI/cert) — DoS/PortScan ยังทำงาน (layer 4)
   ทางแก้ถ้าต้องการ: TLS interception (reverse proxy ถอดรหัส) หรืออ่าน
   access log ของ web server แทน
2. **POST body**: โค้ดอ่าน `request_body` ได้ แต่ Suricata ต้องตั้ง
   `request-body-limit > 0` ใน suricata.yaml ก่อน (default = ไม่ log body)
3. **Switched network**: เครื่องที่รัน Suricata เห็นแค่ traffic ของตัวเอง +
   broadcast/multicast — ต้องทำ port mirroring / รันบน gateway / inline bridge
   ถึงจะเห็นทั้ง network
4. **DoS cold-start**: บริการที่โดนโจมตีตั้งแต่วินาทีแรกโดยไม่มี traffic ปกติ
   มาก่อน → prior+guard ช่วยได้ระดับหนึ่ง แต่ profiling (รันเรียน normal ก่อน)
   ดีที่สุด → ใช้ persist baseline
5. **WebAttack flow-ML**: ปิดเพราะ precision ต่ำ (XSS/SQLi ระดับ flow หายาก
   ใน CICIDS 0.044%) — ใช้ payload ML แทน
6. **Low-and-slow DoS แบบกระจาย** ที่ทุก flow เหมือน benign ทุกประการ — ไม่มี
   NIDS ตัวเดียวจับได้ (ข้อจำกัดเชิงทฤษฎี)

---

## 9. งานที่ทำต่อได้ (ถ้ามีเวลา/อยากพัฒนา)

1. **Suricata config บน network จริง** — เปิด POST body, ทำ port mirroring,
   ตรวจว่าเห็น traffic เครื่องอื่นไหม (`jq 'select(.event_type=="flow")|.src_ip'`)
2. **จูน DoS baseline บน traffic จริง** — ปล่อยเรียน 10-30 นาที ก่อนทดสอบ
   โจมตี (persist ลง dos_baseline.json)
3. **Session-level features** (ทาง A) — รวม flow เป็น session ต่อ src ดู
   pattern (4xx count, URL diversity, path entropy) จับ web scanner (dirb/gobuster)
4. **Windowed supervised ML** (ทาง B, ดีสุด) — ต้องหา CICIDS เวอร์ชัน
   `GeneratedLabelledFlows` ที่มี Source IP + Timestamp เพื่อสร้าง windowed
   feature (connection rate ต่อ src) แล้วให้ ML เรียน threshold เอง = แทน
   adaptive rate ด้วย ML แท้
5. **Payload ML robustness** — เพิ่ม random URL เป็น benign ในชุดเทรน payload
   เพื่อลด FP จาก GoldenEye random path (ตอนนี้แก้ด้วย cooldown แล้ว)
6. วัด detection rate เป็น % (ยิง XSS/SQLi หลาย payload) ทำตารางในเล่ม

---

## 10. ไฟล์สำคัญในโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| `hybrid_nids.py` | ตัวหลัก (train + realtime + all detection) |
| `train_payload.py` | เทรน payload ML (XSS/SQLi) แยกต่างหาก — **ต้องรันเอง** ชี้ model_dir เดียวกัน |
| `build_web_payload_dataset.py` | สร้าง dataset สำหรับ payload ML |
| `--verify-live` (โหมดใน hybrid_nids) | ตรวจ train/serve skew ก่อนเทรน |
| `compare_train_live.py` | เทียบ feature เทรน vs live |
| `check_cicids_flags.py` | ตรวจค่าดิบใน CICIDS |
| `debug_flow_scores.py` | ดูคะแนน ML ต่อ flow |
| `selftest_models.py` | ตรวจโมเดลก่อนรันจริง |
| `model/` | โมเดล .pkl + features.json + thresholds*.json + dos_baseline.json |

**ต้นเหตุที่ payload ML เคยไม่โหลด**: เทรน flow model กับ payload model คนละ
เครื่อง/โฟลเดอร์ — ต้องอยู่ `model_dir` เดียวกัน (payload_vectorizer.pkl,
payload_xgb.pkl, payload_meta.json อยู่ข้าง portscan/ dos/ bruteforce/)

---

## 11. คำสั่งที่ใช้บ่อย

```bash
# Suricata (ตัวเดียว, ไม่มี process ซ้ำ) — บน lab ใช้ BPF filter
sudo systemctl stop suricata && sudo systemctl disable suricata
sudo pkill -9 suricata && sudo rm -f /run/suricata.pid
sudo suricata -c /etc/suricata/suricata.yaml -i enp0s8 -D \
    "host 192.168.1.52 or host 192.168.1.54"    # BPF = arg ท้าย ไม่ใช่ --bpf-filter

# ตรวจก่อนเทรน
python3 hybrid_nids.py --verify-live /var/log/suricata/eve.json

# เทรน
python3 hybrid_nids.py --train <DATASET> --model_dir model
python3 train_payload.py web_payloads_merged.csv --model_dir model

# รันจริง (เร็ว)
python3 hybrid_nids.py --realtime /var/log/suricata/eve.json --xgb_only

# ทดสอบโจมตีจาก Kali (192.168.1.54)
nmap -sS 192.168.1.52                              # PortScan
hydra -l root -P wordlist ssh://192.168.1.52       # BruteForce
python3 goldeneye.py http://192.168.1.52:8080/ -w 10 -s 10 -m get  # DoS
curl "http://192.168.1.52/vuln/sqli/?id=1' OR '1'='1"             # SQLi (HTTP only)
```

---

## 12. กฎการแก้โค้ด (กันพลาดซ้ำ)

1. **แก้แล้วต้องทดสอบด้วยข้อมูลจริง/จำลอง** ไม่ใช่แค่ py_compile — โปรเจกต์นี้
   บั๊กส่วนใหญ่คือ logic ที่ compile ผ่านแต่ทำงานผิด
2. **temporal detector ใช้ event-time เสมอ** (`_event_time(raw)`)
3. **อย่าเพิ่ม feature ที่ live คำนวณไม่ได้** — รัน `--verify-live` ยืนยันก่อน
4. **DoS = adaptive rate เท่านั้น** อย่าเอา hardcode count กลับมา และอย่าให้
   per-flow ML ยิง DoS/PortScan (unreliable)
5. **ทดสอบ cold-start** ทุกครั้งที่แตะ DoS detector (โจมตีตั้งแต่ event แรก
   โดยไม่มี warmup)
6. หลังแก้ ควรผ่าน regression: nmap→PortScan, hydra→BruteForce, GoldenEye→DoS,
   sqlmap ไม่เป็น PortScan, เว็บปกติ/page-load ไม่ FP, nmap ไม่เป็น DoS

---

*อัปเดตล่าสุด: v5.9 — DoS ตรวจด้วย adaptive rate, PortScan/DoS per-flow ML
ปิด, cold-start แก้แล้ว, event-time ใช้ทุก temporal layer, baseline persist
ข้ามรอบ*
