# ตั้งค่า Suricata สำหรับ SWORD (hybrid_nids.py --realtime)

`monitor_realtime()` อ่าน `eve.json` ของ Suricata แบบ real-time แล้วส่งเข้า
`_process_line()` — ต้องมี event 3 ชนิดถึงจะครบทุก pipeline: `flow` (DoS/
PortScan/BruteForce ผ่าน flow ML), `http` (signature → payload-ML → flow ML
สำหรับ XSS/SQLi/BruteForce), `alert` (ถ้าอยากได้ signature ของ Suricata เอง
ด้วย เป็น bonus ไม่ใช่ตัวหลักของระบบนี้)

## 1) ติดตั้ง (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install suricata -y
suricata --version   # แนะนำ 6.x ขึ้นไป
```

## 2) แก้ `/etc/suricata/suricata.yaml`

### 2.1 เปิด event type ให้ครบ (จุดที่พลาดบ่อยที่สุด)

หลาย distro เปิดมาแค่ `alert`/`dns`/`tls`/`http` โดย **ไม่มี `flow`** — ถ้าไม่
เปิด `flow` จะไม่มี event ให้ DoS/PortScan/BruteForce ทำงานเลย:

```yaml
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: eve.json
      types:
        - flow          # ต้องมี — DoS/PortScan/BruteForce ใช้ event นี้
        - http:
            extended: yes
        - alert
```

### 2.2 เปิดอ่าน POST body (สำคัญกับ SQLi/XSS ที่ซ่อนใน POST)

**อัปเดต (v5.10):** `request-body-limit > 0` เพียงอย่างเดียว **ไม่พอ** —
พิสูจน์แล้วบนเครื่องจริงว่า `event_type=="http"` ของ Suricata ไม่มีช่องให้
body เลยไม่ว่าจะตั้งค่านี้สูงแค่ไหนก็ตาม (ค่านี้ควบคุมแค่ความลึกการ
reassemble ภายในสำหรับ signature matching ไม่ใช่การ log body ออก eve.json)
วิธีเดียวที่ทำให้ Suricata แนบ body มาด้วยได้คือให้ signature rule ยิง
alert แล้วเปิด `http-body-printable` ใต้ eve-log **"alert"** (คนละที่กับ
"http") ต้องทำครบ 3 ขั้นตอนนี้:

**(ก) ตั้ง request-body-limit ตามปกติ** (ยังจำเป็น — ควบคุมว่า Suricata
reassemble body ลึกแค่ไหนก่อนจะมีอะไรให้ dump):
```yaml
app-layer:
  protocols:
    http:
      libhtp:
        default-config:
          request-body-limit: 100kb   # หรือค่าที่ต้องการ ต้อง > 0
```

**(ข) เปิด `http-body-printable` + `payload-printable` ใต้ eve-log "alert"**
(ไม่ใช่ใต้ "http"):
```yaml
outputs:
  - eve-log:
      types:
        - alert:
            payload-printable: yes    # fallback source
            http-body-printable: yes  # ต้อง requires metadata (default yes อยู่แล้ว)
```

**(ค) เพิ่ม custom rule ที่ยิง alert ทุก POST request** เพื่อให้มี alert
event ให้แนบ body ด้วย (ไม่งั้นไม่มี alert เกิดขึ้นเลยสำหรับ POST ปกติ
ที่ไม่ตรง signature จริงตัวไหน) — ใช้ `suricata_rules/sword-local.rules`
ที่มากับโปรเจกต์นี้:
```yaml
rule-files:
  - suricata.rules
  - /path/to/SWORD/suricata_rules/sword-local.rules
```

**สำคัญ:** rule นี้ยิง alert ทุก POST จริง แต่ **ไม่ใช่การแจ้งเตือนโจมตี
โดยตรง** — `hybrid_nids.py` (`_handle_suricata_alert` →
`_handle_body_capture_alert`) ดัก `sid=9000001` (=
`SWORD_BODY_CAPTURE_SID` ในโค้ด) เป็นกรณีพิเศษ เอา body ไปวิ่งผ่าน
signature override + payload ML ก่อน ถึงจะยิง alert จริงถ้าเจอ payload
— POST ปกติ (login ทั่วไป ฯลฯ) จะไม่ถูกแจ้งเตือน มี overhead แค่ alert
event เพิ่ม 1 อันต่อ POST request 1 ครั้งเท่านั้น

### 2.3 ตั้ง interface ที่จะดักฟัง (แก้ตาม NIC จริงของเครื่อง)

```yaml
af-packet:
  - interface: eth0          # เปลี่ยนเป็น interface จริง (ดูด้วย `ip a`)
    cluster-id: 99
    cluster-type: cluster_flow
```

## 3) (แนะนำ) โหลด ruleset จริงไว้ด้วย เผื่ออยากได้ signature ของ Suricata เองมาเทียบ

```bash
sudo suricata-update
```

## 4) รัน Suricata

```bash
sudo suricata -c /etc/suricata/suricata.yaml -i eth0 -D
# ตรวจว่ามี event ไหลจริง
sudo tail -f /var/log/suricata/eve.json
```

## 5) สิทธิ์อ่านไฟล์ (สำคัญกับ user ธรรมดา เช่นตอนทดสอบกับ Kali)

`/var/log/suricata/eve.json` ปกติเป็นของ root — ถ้ารัน `hybrid_nids.py
--realtime` ด้วย user ธรรมดา (ไม่ sudo) จะอ่านไฟล์ไม่ได้ เลือกทางใดทางหนึ่ง:

```bash
# ทางที่ง่ายที่สุด — รันตัว python เองด้วย sudo
sudo venv/bin/python hybrid_nids.py --realtime /var/log/suricata/eve.json --model_dir model

# หรือเพิ่ม user เข้ากลุ่มที่อ่านไฟล์ suricata ได้ (ถ้า suricata สร้างกลุ่มไว้)
sudo usermod -aG suricata $USER   # แล้ว logout/login ใหม่
```

`hybrid_nids.py` เองมี fallback อยู่แล้วสำหรับ log ของตัวเอง (ถ้าเขียน
`/var/log/sword_detection` ไม่ได้จะ fallback มาเขียนที่
`./sword_detection_logs` แทนอัตโนมัติ) แต่การ**อ่าน** `eve.json` ของ
Suricata ยังต้องมีสิทธิ์อ่านไฟล์นั้นอยู่ดี แก้ไม่ได้จากฝั่งสคริปต์นี้

## 6) รัน SWORD

```bash
python hybrid_nids.py --realtime /var/log/suricata/eve.json --model_dir model
```

ถ้าไฟล์ยังไม่มี (Suricata ยังไม่ทันสร้าง) สคริปต์จะรอเฉย ๆ ไม่ error ทันที
กด Ctrl+C เพื่อหยุดได้ทุกเมื่อ จะสรุปจำนวน event ที่ประมวลผลไปก่อนออก

## 7) ทดสอบด้วย Kali

จำลอง traffic จากเครื่อง Kali ไปยังเครื่องเป้าหมายที่ Suricata ดักฟังอยู่
(ต้องอยู่ใน network segment เดียวกับ interface ที่ตั้งไว้ในข้อ 2.3) — ตัวอย่าง
เช่น:

- Brute Force: `hydra -l admin -P rockyou.txt <target-ip> http-post-form "..."` หรือ `hydra ssh://<target-ip>`
- SQLi/XSS: ส่ง HTTP request ที่มี payload ใน URL/POST ไปยังเว็บเป้าหมาย (เช่น
  ด้วย `curl` หรือ Burp Suite) — จะโดนดักตั้งแต่ signature override
  (`detect_web_signature()`) ถ้า payload ตรงกับ pattern ที่รู้จัก หรือหลุดไป
  ให้ payload-ML ตัดสินต่อถ้า payload ไม่ตรง pattern ตรง ๆ
- PortScan: `nmap -sS <target-ip>`

ดู alert แบบ real-time จาก stdout ของสคริปต์ หรือดู log ที่ `sword_detection_logs/ml_log.json` (หรือ `/var/log/sword_detection/ml_log.json` ถ้ารันด้วย sudo)
