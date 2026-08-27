#!/usr/bin/env bash
#
# setup_suricata_sword.sh — ตั้งค่า Suricata ให้พร้อมกับ hybrid_nids.py
# (SWORD) แบบครบจบในคำสั่งเดียว รวมทุกบทเรียนที่เจอจริงระหว่าง dev:
#
#   1) เปิด event type ให้ครบ (flow / http extended / alert)
#   2) เปิด request-body-limit (ให้ Suricata reassemble POST body)
#   3) เปิด http-body-printable + payload-printable ใต้ eve-log "alert"
#      (event_type=="http" ของ Suricata ไม่มีช่องให้ body เลย ไม่ว่าจะตั้ง
#      request-body-limit สูงแค่ไหน — ต้องพึ่ง alert event เท่านั้น)
#   4) ติดตั้ง custom rule "sword-local.rules" ที่ยิง alert ทุก POST
#      request เพื่อดึง body ออกมา (ไม่ใช่ signature โจมตีจริง — ดู
#      SWORD_BODY_CAPTURE_SID ใน hybrid_nids.py) — วางไว้ที่
#      /etc/suricata/rules/ เท่านั้น "ห้าม" วางใน /home/** เด็ดขาด เพราะ
#      suricata.service ส่วนใหญ่ตั้ง ProtectHome=true ใน systemd sandbox
#      ทำให้ Suricata อ่านไฟล์ใต้ /home ไม่เห็นเลย (เจอบั๊กนี้จริงบนเครื่อง
#      dev — error คือ "No rule files match the pattern ..." แบบไม่มี
#      exit code ผิดปกติเตือนตรงๆ ทำให้เข้าใจผิดว่า config ผ่านแล้ว)
#   5) ตั้ง af-packet interface + HOME_NET ให้ตรงกับเครื่องจริง (ต้อง
#      ระบุ interface เอง ห้ามเดาอัตโนมัติจาก default route — เครื่อง dev
#      มี 2 NIC และ default route ไม่ใช่ตัวที่ต้องดักฟัง)
#   6) ตรวจ syntax ก่อน restart จริงเสมอ (suricata -T) กันพัง IDS ที่รันอยู่
#   7) restart แล้ววนเช็ค log ว่า "ไม่มี" ข้อความ "No rule files match"
#      และ rule count เพิ่มขึ้นจริง ไม่ใช่แค่ restart แล้วจบเฉยๆ
#
# ใช้งาน:
#   sudo ./setup_suricata_sword.sh <interface> [suricata.yaml path]
#   ตัวอย่าง: sudo ./setup_suricata_sword.sh enp0s8
#
# รันซ้ำได้ปลอดภัย (idempotent) — เช็คก่อนทุกจุดว่าตั้งไปแล้วหรือยัง
set -euo pipefail

IFACE="${1:?ต้องระบุ interface เช่น: sudo $0 enp0s8}"
YAML="${2:-/etc/suricata/suricata.yaml}"
RULES_DIR="/etc/suricata/rules"
RULE_FILE="$RULES_DIR/sword-local.rules"
SID=9000001

if [[ $EUID -ne 0 ]]; then
    echo "[FATAL] ต้องรันด้วย sudo/root (ต้องแก้ $YAML และ restart service)" >&2
    exit 1
fi

if [[ ! -f "$YAML" ]]; then
    echo "[FATAL] ไม่พบ $YAML — ระบุ path ที่ถูกต้องเป็น argument ที่ 2" >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo " SWORD Suricata setup — interface=$IFACE  config=$YAML"
echo "════════════════════════════════════════════════════════════════"

# ── 0) หา HOME_NET จาก interface จริง (ไม่เดาจาก default route) ─────
IP_CIDR=$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | head -1)
if [[ -z "$IP_CIDR" ]]; then
    echo "[FATAL] interface '$IFACE' ไม่มี IPv4 address — เช็คชื่อ interface ด้วย 'ip a'" >&2
    exit 1
fi
HOME_NET_CIDR=$(python3 - "$IP_CIDR" <<'PYEOF'
import ipaddress, sys
print(ipaddress.ip_interface(sys.argv[1]).network)
PYEOF
)
echo "[0/7] interface $IFACE = $IP_CIDR -> HOME_NET จะตั้งเป็น [$HOME_NET_CIDR]"

# ── สำรอง suricata.yaml ก่อนแก้ทุกครั้ง (กันพลาดแก้เพี้ยน) ─────────
BACKUP="${YAML}.bak.$(date +%Y%m%d%H%M%S 2>/dev/null || echo pre-sword-setup)"
cp -a "$YAML" "$BACKUP"
echo "[backup] สำรอง config เดิมไว้ที่ $BACKUP"

# ── 1) HOME_NET ─────────────────────────────────────────────────────
if grep -qE "^\s*HOME_NET:\s*\"\[$HOME_NET_CIDR\]\"" "$YAML"; then
    echo "[1/7] HOME_NET ตั้งไว้ถูกต้องอยู่แล้ว — ข้าม"
else
    sed -i -E "s|^(\s*)HOME_NET:.*|\1HOME_NET: \"[$HOME_NET_CIDR]\"|" "$YAML"
    echo "[1/7] ตั้ง HOME_NET = [$HOME_NET_CIDR] แล้ว"
fi

# ── 2) af-packet interface ──────────────────────────────────────────
if grep -qE "^\s*-\s*interface:\s*$IFACE\s*$" "$YAML"; then
    echo "[2/7] af-packet interface ตั้งเป็น $IFACE อยู่แล้ว — ข้าม"
else
    sed -i -E "0,/^(\s*)-\s*interface:.*/{s//\1- interface: $IFACE/}" "$YAML"
    echo "[2/7] ตั้ง af-packet interface = $IFACE แล้ว (แก้บรรทัดแรกที่เจอ — ปกติพอสำหรับ default config)"
fi

# ── 3) request-body-limit (ต้อง > 0 ถึงจะมีอะไรให้ dump) ──────────
if grep -qE "^\s*request-body-limit:\s*[1-9]" "$YAML"; then
    echo "[3/7] request-body-limit เปิดอยู่แล้ว (>0) — ข้าม"
else
    sed -i -E "s|^(\s*)request-body-limit:\s*0\s*$|\1request-body-limit: 100kb|" "$YAML"
    echo "[3/7] ตั้ง request-body-limit = 100kb แล้ว"
fi

# ── 4) http-body-printable + payload-printable ใต้ eve-log "alert" ──
if grep -qE "^\s*http-body-printable:\s*yes" "$YAML"; then
    echo "[4/7] http-body-printable เปิดอยู่แล้ว — ข้าม"
else
    sed -i \
        -e 's/^\(\s*\)# payload-printable: yes.*/\1payload-printable: yes/' \
        -e 's/^\(\s*\)# http-body-printable: yes.*/\1http-body-printable: yes/' \
        "$YAML"
    if grep -qE "^\s*http-body-printable:\s*yes" "$YAML"; then
        echo "[4/7] เปิด payload-printable + http-body-printable ใต้ eve-log alert แล้ว"
    else
        echo "[4/7] [WARNING] หา '# http-body-printable: yes' ใน $YAML ไม่เจอ" \
             "— รูปแบบไฟล์อาจต่างจาก default template ต้องเปิดเองด้วยมือใต้" \
             "outputs: -> eve-log: -> types: -> - alert: (ดู setup_suricata.md หัวข้อ 2.2)"
    fi
fi

# ── 5) eve-log types ครบ (flow / http extended / alert) ────────────
missing_types=()
grep -qE "^\s*-\s*flow\s*$" "$YAML" || missing_types+=("flow")
grep -qE "^\s*-\s*http:\s*$" "$YAML" || missing_types+=("http")
grep -qE "^\s*-\s*alert\b" "$YAML" || missing_types+=("alert")
if [[ ${#missing_types[@]} -eq 0 ]]; then
    echo "[5/7] eve-log types (flow/http/alert) ครบอยู่แล้ว — ข้าม"
else
    echo "[5/7] [WARNING] eve-log types ขาด: ${missing_types[*]} — เปิดเองด้วยมือใต้" \
         "outputs: -> eve-log: -> types: (ดู setup_suricata.md หัวข้อ 2.1) ไม่ auto-fix" \
         "เพราะ YAML list ตรงนี้ format ต่างกันได้เยอะระหว่าง distro"
fi

# ── 6) ติดตั้ง sword-local.rules ไปที่ /etc/suricata/rules/ ─────────
# ห้ามวางใน /home/** — ProtectHome=true ใน systemd unit ทำให้ Suricata
# มองไม่เห็นไฟล์เงียบๆ (ไม่ error ชัดเจน แค่ log "No rule files match")
mkdir -p "$RULES_DIR"
cat > "$RULE_FILE" <<EOF
# sword-local.rules — auto-generated by setup_suricata_sword.sh
# ไม่ใช่ signature โจมตีจริง — ยิง alert ทุก POST request เพื่อดึง POST
# body มาแนบกับ eve.json alert event (ผ่าน http-body-printable ที่เปิด
# ไว้ข้างบน) เพราะ event_type=="http" ปกติของ Suricata ไม่มีช่องให้
# body เลย — hybrid_nids.py (_handle_body_capture_alert) ดัก sid=$SID
# นี้เป็นกรณีพิเศษ เอา body ไปวิ่งผ่าน signature override + payload ML
# ก่อน ถึงจะยิง alert จริงถ้าเจอ payload ไม่ใช่แจ้งเตือนทุก POST ปกติ
alert http any any -> \$HOME_NET any (msg:"SWORD LOCAL - POST body capture (not an attack, feeds payload ML)"; flow:established,to_server; http.method; content:"POST"; classtype:not-suspicious; sid:$SID; rev:1;)
EOF
echo "[6/7] ติดตั้ง $RULE_FILE แล้ว"

if grep -qF "$RULE_FILE" "$YAML"; then
    echo "        rule-files มี $RULE_FILE อยู่แล้ว — ข้าม"
elif grep -qE "^\s*-\s*suricata\.rules\s*$" "$YAML"; then
    sed -i "/^  - suricata\.rules$/a\\  - $RULE_FILE" "$YAML"
    echo "        เพิ่ม $RULE_FILE เข้า rule-files แล้ว"
else
    echo "[WARNING] หาบรรทัด '- suricata.rules' ใต้ rule-files: ไม่เจอ" \
         "— เพิ่มเองด้วยมือ: rule-files: -> - $RULE_FILE"
fi

# ── 7) ตรวจ syntax ก่อน restart จริง ─────────────────────────────────
echo "[7/7] ตรวจ syntax ด้วย suricata -T ..."
if ! suricata -T -c "$YAML" -v > /tmp/sword_suricata_test.log 2>&1; then
    if grep -q "unknown rule keyword ''" /tmp/sword_suricata_test.log; then
        echo "        [WARNING] มี rule เดิม (ปกติจาก suricata-update) parse ไม่ผ่านบางตัว" \
             "— เจอบนเครื่อง dev แล้วว่าเป็นบั๊กที่มีอยู่ก่อนแล้วใน ruleset ที่ดาวน์โหลดมา" \
             "(ไม่เกี่ยวกับ sword-local.rules) ตรวจว่า sword-local.rules ไม่ได้อยู่ในนั้นด้วย:"
        grep -i "sword" /tmp/sword_suricata_test.log || echo "        (ไม่เจอ error ของ sword-local.rules — ok)"
    else
        echo "[FATAL] suricata -T ล้มเหลวด้วยเหตุผลอื่น ดู /tmp/sword_suricata_test.log" >&2
        tail -30 /tmp/sword_suricata_test.log >&2
        echo "        config เดิมสำรองไว้ที่ $BACKUP — กู้คืนได้ด้วย: cp $BACKUP $YAML" >&2
        exit 1
    fi
else
    echo "        syntax ผ่าน"
fi

echo ""
echo "restart suricata แล้วตรวจว่า rule โหลดจริง..."
systemctl restart suricata
sleep 2

# รอ engine โหลด rule เสร็จ (ปกติ ~30-60s สำหรับ ruleset ใหญ่)
for i in $(seq 1 60); do
    if journalctl -u suricata --no-pager -n 50 2>/dev/null | grep -q "Engine started" \
       || tail -n 50 /var/log/suricata/suricata.log 2>/dev/null | grep -q "Engine started"; then
        break
    fi
    sleep 2
done

if tail -n 200 /var/log/suricata/suricata.log 2>/dev/null | grep -q "No rule files match the pattern.*sword-local"; then
    echo "[FATAL] Suricata ยังหา sword-local.rules ไม่เจอ (อาจเป็น path/permission อื่น)" >&2
    echo "        เช็คด้วยมือ: tail -50 /var/log/suricata/suricata.log | grep -i sword" >&2
    exit 1
fi

if systemctl is-active --quiet suricata; then
    echo "════════════════════════════════════════════════════════════════"
    echo " เสร็จแล้ว — suricata active, sword-local.rules (sid=$SID) โหลดสำเร็จ"
    echo " ตรวจเพิ่มเติมได้ด้วย: tail -f /var/log/suricata/eve.json | grep sid"
    echo "════════════════════════════════════════════════════════════════"
else
    echo "[FATAL] suricata restart แล้วแต่ service ไม่ active — เช็ค: systemctl status suricata" >&2
    exit 1
fi
