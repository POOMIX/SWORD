"""
hybrid_nids.py — OvR Binary Hybrid NIDS (DoS / PortScan / BruteForce /
Benign runtime; WebAttack ยังเทรนไว้แต่ไม่ใช้ตัดสิน alert แล้ว)
v3.6 — แก้บั๊ก encoding ที่ทำข้อมูลหาย + ปิด WebAttack flow signal ตอน
inference (แม่นไม่พอ ให้ payload-ML คุมแทน)
ใช้ ML ensemble: Decision Tree + Random Forest + XGBoost

Changelog v3.6:
- [FIX] บั๊ก encoding ร้ายแรง: ไฟล์ Thursday-WorkingHours-Morning-
  WebAttacks.pcap_ISCX.csv (CICIDS2017) export มาไม่ใช่ UTF-8 แท้ ทำให้
  en-dash ใน "Web Attack – XSS"/"– Brute Force"/"– Sql Injection" decode
  เพี้ยนเป็น U+FFFD ตอน pandas อ่านแบบ utf-8 (default) — LABEL_MAP แบบ
  เทียบ string ตรง ๆ เลย "ไม่ match เลยสักแถวเดียว" ทั้งไฟล์ (2,180 rows
  หายไปแบบไม่มีใครรู้ตัว มาตั้งแต่ก่อน BruteForce จะถูกแยกออกด้วยซ้ำ)
  แก้ด้วย _map_label() ที่จับ keyword แทนเทียบ string ทั้งดุ้น ยืนยันผล
  แล้ว: WebAttack 317→990 rows, BruteForce +1,470 rows
- [CHANGE] ปิด "webattack" ออกจาก _RUNTIME_FLOW_TASKS ตอน inference:
  แม้แก้บั๊ก encoding แล้ว WebAttack (XSS+SQLi ระดับ flow) ยังมีแค่ 990
  rows จาก 2.25M (0.044%) — held-out test precision=0.083 (false
  positive ~11 ครั้งต่อ true positive 1 ครั้ง) ไม่พอสำหรับตัดสินใจ alert
  จริง ตัวจับ XSS/SQLi หลักตอนนี้คือ payload-ML pipeline (TF-IDF+XGBoost
  — ยังเป็น ML เหมือนเดิม แค่คนละ pipeline, F1 0.98-0.99 ทั้ง
  in-distribution และ OOD) + signature override ที่มีอยู่แล้ว
  train_models() ยังเทรน+เซฟโมเดล webattack ไว้เผื่ออ้างอิงในเล่มจบ
  แค่ HybridNIDS._load_models() ไม่โหลดเข้ามาตัดสินใจ alert เท่านั้น

Changelog v3.5:
- [NEW] Brute Force แยกเป็น OvR task ของตัวเอง ("bruteforce") ไม่ merge
  รวมเข้า WebAttack แล้ว — เทรนจาก CICIDS2017 Web Attack file (web login
  brute force) + Tuesday-WorkingHours.pcap_ISCX.csv (FTP/SSH-Patator) +
  CICIDS2018 Thursday-22/Friday-23-02-2018 (Brute Force -Web) +
  Wednesday-14-02-2018 (FTP-BruteForce/SSH-Bruteforce)
- [FIX] เดิม flow-model ทาย "WebAttack" (binary, merge Brute
  Force+XSS+SQLi ไว้ด้วยกัน) แล้ว _analyze_flow/_analyze_http_event เดา
  เอาเองว่า specific_type = "Brute Force Attack" เสมอ (เพราะ XSS/SQLi
  ถูกดักด้วย signature/payload-ML ไปก่อนแล้ว) — เป็น heuristic ไม่ใช่การ
  เรียนรู้จริง ตอนนี้ BruteForce มีโมเดลเทรนแยกของตัวเอง ใช้
  _PREDICTED_CLASS_DISPLAY แสดงผลตรงจาก predicted_class ของ ensemble
  แทน ถ้า flow-model ยังทาย WebAttack (ไม่ใช่ BruteForce) แปลว่าเป็น
  XSS/SQLi ระดับ flow ที่ signature/payload ยังจับไม่ได้ รายงานเป็น
  "unspecified subtype" ตรง ๆ แทนการเดาผิดประเภท

Changelog v3.4:
- [FIX] เอา softmax cross-task normalization ออก: เดิม threshold ถูก tune
  บน raw ensemble score แต่ predict() เทียบกับ softmax score คนละสเกล
  ทำให้ traffic ปกติที่ 3 task มีคะแนนใกล้เคียงกันเสี่ยงโดน flag ผิด
  (softmax ดันค่าขึ้นไปใกล้ 1/3 ได้แม้ raw score จะต่ำมาก) และ attack จริง
  ที่คะแนนปานกลางเสี่ยงหลุด threshold ไป (false negative) — ตอนนี้ใช้
  raw score เทียบ threshold ตรงๆ เหมือนตอน tune
- [FIX] has_suspicious_chars (XSS/SQLi pattern) เดิมเป็น ML feature ที่
  train มา constant=0 เสมอ (CICIDS ไม่มี column นี้) → โมเดลไม่เคยเรียนรู้
  ที่จะใช้มันได้เลย (variance=0 = information gain=0) ตอนนี้เปลี่ยนเป็น
  signature-based override ที่ bypass ML ไปเลยเมื่อเจอ pattern ที่รู้จักแน่ๆ
  ส่วน ML ยังทำหน้าที่จับ Brute Force ต่อ (behavioral, จับได้จริงจาก
  bytes-ratio/flow features)
- [NEW] specific_type ใน alert: XSS, SQL Injection, Brute Force
- [NEW] URI decoding ก่อนตรวจ signature (ป้องกัน encoded payload)
- [FIX] CICIDS2018 column name mapping + numeric coercion
- [FIX] "Benign" casing (CICIDS2018 ใช้ "Benign" ไม่ใช่ "BENIGN")
- [FIX] Train/Val/Test split 3 ทาง: val ใช้ tune threshold, test ใช้ report
- [FIX] Multi-file dataset loading (2018 WebAttack กระจายหลายไฟล์)
"""

VERSION = "5.13-post-body-capture"

import os
import json
import time
import signal
import ipaddress
import threading
import queue
import logging
import argparse
import pickle
from pathlib import Path
from urllib.parse import unquote, parse_qsl
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  กลุ่ม Label ที่สนใจ
# ─────────────────────────────────────────────
LABEL_MAP = {
    # CICIDS2017 ใช้ "BENIGN" (ตัวพิมพ์ใหญ่หมด)
    "BENIGN": "Benign",
    # CICIDS2018 ใช้ "Benign" (ตัวพิมพ์เล็กปกติ) — ถ้าไม่แมป traffic ปกติ
    # จาก 2018 จะโดน dropna() ทิ้งทั้งหมด
    "Benign": "Benign",
    # DoS / DDoS
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    # PortScan
    "PortScan": "PortScan",
    "Port Scan": "PortScan",
    # Web Attacks (CICIDS2017) — เหลือแค่ XSS/SQLi ที่นี่ (payload-based,
    # ให้ Payload ML ที่ train_payload.py แยกย่อยต่อ) ส่วน "Brute Force"
    # แยกออกเป็น task ของตัวเองแล้ว (ดูด้านล่าง) ไม่ merge เข้า WebAttack
    # อีกต่อไป เพราะเดิมทำให้ flow-model แยก Brute Force ไม่ได้จริง ต้อง
    # เดา specific_type เอาเองตอน inference (v3.4 fallback heuristic)
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack – SQL Injection": "WebAttack",
    "Web Attack – XSS + SQL Injection": "WebAttack",
    # Web Attacks (CICIDS2018)
    "Brute Force -XSS": "WebAttack",  # ชื่อ CICIDS2018 เรียก "Brute Force" แต่จริง ๆ คือ XSS
    "SQL Injection": "WebAttack",
    # Brute Force — task แยกของตัวเอง (v3.5): รวมทั้ง web-login brute
    # force (HTTP form) และ network-service brute force (FTP/SSH) เพราะ
    # ทั้งคู่มีลักษณะ flow เดียวกันคือ "ยิง request/connection ซ้ำถี่ ๆ"
    "Web Attack – Brute Force": "BruteForce",       # CICIDS2017 (web login)
    "Brute Force -Web": "BruteForce",                # CICIDS2018 (web login)
    "FTP-Patator": "BruteForce",                     # CICIDS2017 (network)
    "SSH-Patator": "BruteForce",                     # CICIDS2017 (network)
    "FTP-BruteForce": "BruteForce",                  # CICIDS2018 (network)
    "SSH-Bruteforce": "BruteForce",                  # CICIDS2018 (network)
}

TARGET_CLASSES = {"Benign", "DoS", "PortScan", "WebAttack", "BruteForce"}


def _map_label(raw: str) -> str:
    """แปลง raw Label → target class เดียวกับ LABEL_MAP แต่กันบั๊ก encoding
    ที่เจอจริง: ไฟล์ CICIDS2017 Thursday-WorkingHours-Morning-WebAttacks
    ไม่ได้ export เป็น UTF-8 แท้ ๆ ตัว en-dash (–) ใน "Web Attack – XSS"/
    "Web Attack – Brute Force"/"Web Attack – Sql Injection" เลย decode
    เพี้ยนเป็น U+FFFD ("Web Attack � XSS") ตอน pandas.read_csv() อ่านแบบ
    utf-8 (default) — ผลคือ LABEL_MAP.get() แบบเทียบ string ตรง ๆ ไม่ match
    เลยสักแถวเดียว (ยืนยันแล้วด้วยการเช็คไฟล์จริง: ทั้ง 3 คลาส Web Attack
    ของไฟล์นี้ = 2,180 rows หายไปหมดจากทั้ง WebAttack และ BruteForce task
    แบบไม่มีใครรู้ตัว — ก่อนหน้านี้แม้แต่ตอนที่ Brute Force ยัง merge เข้า
    WebAttack ก็โดนบั๊กนี้เหมือนกัน)
    แก้ด้วยการจับ keyword แทนเทียบ string ทั้งดุ้น ไม่สนว่า dash character
    ตรงกลางจะถูก decode ออกมาเป็นอะไร"""
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    if raw.startswith("Web Attack"):
        low = raw.lower()
        if "brute force" in low:
            return "BruteForce"
        if "xss" in low:
            return "WebAttack"
        if "sql injection" in low:
            return "WebAttack"
    return None


# ─────────────────────────────────────────────
#  ML Constants
# ─────────────────────────────────────────────
MIN_FLOW_DURATION_S: float = 1e-6

# Default ensemble weights — จะถูกแทนที่ด้วย F1-based weights
# จาก weights.json ที่ train_models() สร้างไว้
DT_WEIGHT:  float = 0.15
RF_WEIGHT:  float = 0.30
XGB_WEIGHT: float = 0.55

DEFAULT_THRESHOLD: float = 0.5

# เปิด/ปิด signature-override สำหรับ web attack (XSS/SQLi)
# True  = ถ้า payload ตรง _XSS_PATTERNS/_SQLI_PATTERNS -> alert ทันที
#         (เร็ว แต่เป็นกฎ ไม่ใช่ ML)
# False = ให้ payload ML (TF-IDF+XGBoost) เป็นคนตัดสินเว็บล้วน ๆ
#         *** ไม่กระทบ Suricata signature *** ซึ่งเป็นคนละเส้นทาง
#         (_handle_suricata_alert อ่าน event_type=="alert" จาก Suricata)
USE_SIGNATURE_OVERRIDE: bool = False

# v5.3: จำนวน benign สูงสุดที่ดึงจาก "แต่ละไฟล์" มารวมเป็น shared pool
# (คุมหน่วยความจำ + คุมสัดส่วนไม่ให้ไฟล์ใหญ่ครอบงำ benign ทั้งหมด)
PER_FILE_BENIGN_CAP: int = 120_000

# ─────────────────────────────────────────────
#  Suricata signature alert filtering
# ─────────────────────────────────────────────
# บนเครือข่ายจริง ET INFO / policy rules (STUN, DNS, TLS, NTP) ยิง alert
# เยอะมากจาก traffic ปกติ = "alert fatigue" ของ signature IDS (จุดอ่อนที่
# ML แก้ได้ — เขียนเปรียบเทียบในเล่มได้) กรองด้วย "severity" ของ Suricata:
#   1 = สูงสุด (attack ชัด เช่น ET SCAN, ET WEB_ATTACK)
#   2 = medium
#   3 = informational (STUN/DNS/TLS ปกติ — noise)
# แสดงเฉพาะ severity <= SURICATA_MIN_SEVERITY
#   2 = แสดง attack จริง ซ่อน INFO noise  (ค่าแนะนำ)
#   1 = แสดงเฉพาะ severity สูงสุด
#   0 = ปิด Suricata signature alert ทั้งหมด (เหลือแต่ ML ล้วน)
SURICATA_MIN_SEVERITY: int = 0
# บาง rule เป็น severity 2 แต่ก็ยังเป็น noise (Steam, cleartext-pass ปกติ) —
# กรองเพิ่มด้วย "prefix ของชื่อ rule" ที่เป็นกลุ่ม informational/policy
SURICATA_NOISE_PREFIXES: tuple = (
    "ET INFO", "ET POLICY", "ET USER_AGENTS", "ET DNS", "ET TLS",
    "ET JA3", "ET HUNTING", "GPL",
)

# ── POST body capture (v5.10) ───────────────────────────────────────
# eve.json event_type=="http" ของ Suricata "ไม่มีช่องให้ POST body เลย"
# ไม่ว่าจะตั้ง request-body-limit สูงแค่ไหนก็ตาม (ตัวนั้นควบคุมแค่ความลึก
# การ reassemble ภายในสำหรับ signature matching ไม่ใช่การ log ออก
# eve.json) วิธีเดียวที่ทำให้ Suricata แนบ body มาด้วยได้คือให้ signature
# rule ยิง alert แล้วเปิด http-body-printable ใต้ eve-log "alert" (ไม่ใช่
# "http") — จึงต้องมี rule พิเศษ (ดู suricata_rules/sword-local.rules /
# setup_suricata_sword.sh) ที่ยิง alert ทุก POST request เพื่อ "ดึง" body
# ออกมาเท่านั้น ไม่ใช่ signature โจมตีจริง ห้ามให้ _handle_suricata_alert()
# ยิง alert ตรงๆ จาก sid นี้ (ไม่งั้นทุก POST ปกติ เช่น login ทั่วไป จะโดน
# แจ้งเตือนหมด) ต้องดัก sid นี้ไปวิ่งผ่าน signature override + payload ML
# เหมือนกับที่ _analyze_http_event ทำกับ URI/cookie/UA ก่อน ถึงจะยิง alert
# จริงถ้าเจอ — ดู _handle_body_capture_alert() ต้องดักก่อนถึง severity/
# category filter ด้านบน เพราะ category ของ rule นี้คือ "Not Suspicious
# Traffic" ซึ่งโดนกรองทิ้งไปแล้วโดย filter ปกติ
SWORD_BODY_CAPTURE_SID: int = 9000001

# ─────────────────────────────────────────────
#  HOME_NET — เฝ้าระวังเฉพาะเครือข่ายที่เราปกป้อง
# ─────────────────────────────────────────────
# บนเครือข่ายจริง เครื่องภายใน (เช่น Windows host) เปิด connection "ออก
# อินเทอร์เน็ต" หลาย port ตลอดเวลา (เกม, P2P, torrent, streaming) ซึ่งมี
# รูปแบบคล้าย port scan / rate สูง = false positive เพราะนั่นคือ traffic
# ขาออกปกติ ไม่ใช่การโจมตี "เครื่องเรา"
#
# หลักการมาตรฐานของ NIDS (Suricata ก็ใช้ HOME_NET เหมือนกัน): วิเคราะห์
# เฉพาะ flow ที่ "ปลายทาง (dst) อยู่ในเครือข่ายที่เราเฝ้าระวัง" — คือมี
# คนพยายามเข้าถึง/โจมตี asset ของเรา ส่วน local -> internet (ขาออก) ข้าม
# เพราะเราไม่ได้ปกป้องอินเทอร์เน็ต และไม่ควรสอดส่อง traffic ที่ user เปิด
# เว็บออกไปเอง
#
# ค่า default = RFC1918 (private) + IPv6 ULA/link-local ทั้งหมด ปรับให้แคบ
# เป็น subnet เดียว (เช่น "192.168.1.0/24") ได้เพื่อลด FP เพิ่ม
HOME_NET: list = [
    "192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "169.254.0.0/16",
    "fc00::/7", "fe80::/10",
]
_HOME_NET_NETS = []
for _cidr in HOME_NET:
    try:
        _HOME_NET_NETS.append(ipaddress.ip_network(_cidr))
    except ValueError:
        pass


def _ip_in_home_net(ip: str) -> bool:
    """ปลายทางอยู่ในเครือข่ายที่เฝ้าระวังไหม (ดู comment เหนือ HOME_NET)"""
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return any(addr in net for net in _HOME_NET_NETS)

# Secondary rejection: ถ้า score ที่ดีที่สุดใกล้กับอันดับสองเกิน gap
# → ไม่มั่นใจพอ, flag เป็น Benign
SECONDARY_REJECTION_GAP: float = 0.05
# v5.1: ลดจาก 0.15 -> 0.05
# เวอร์ชัน 3.2 ซึ่งเป็นเวอร์ชันที่ตรวจจับ PortScan/DoS ได้จริง ไม่มีกฎนี้
# เลย (ตัดสินด้วย max_score >= threshold อย่างเดียว) การที่โมเดลแยกไม่ออก
# ว่าเป็น attack "ชนิดไหน" ไม่ควรถูกตีความว่า "ไม่ใช่ attack" เพราะคำถาม
# สองข้อนี้คนละคำถามกัน — ยิ่งตอนนี้มี 3 task แข่งกัน (v3.2 มี 2) คะแนน
# ยิ่งใกล้กันง่ายขึ้นมาก และ CalibratedClassifierCV ก็บีบคะแนนเข้าหากลาง
# ทำให้ gap แคบลงอีก สองอย่างรวมกันทำให้ของเดิม 0.15 ตัด attack จริงทิ้ง
# 0.05 ยังกันเคสที่คลุมเครือจริง ๆ ได้ แต่ไม่ทิ้งสิ่งที่ตรวจเจอแล้ว

# ─────────────────────────────────────────────
#  Brute Force Repetition Gate
# ─────────────────────────────────────────────
# BruteForce ตามนิยาม (SSH-Patator/FTP-Patator ใน CICIDS) คือการ "ลอง
# รหัสผ่านซ้ำๆ จำนวนมาก" ไม่ใช่การเชื่อมต่อครั้งเดียว — แต่ในบรรดา
# FEATURES ทั้งหมดที่ใช้เทรน ไม่มี feature ไหนนับ "จำนวนครั้งที่ทำซ้ำใน
# ช่วงเวลาหนึ่ง" (temporal/count aggregation) เลยสักตัว โมเดลจึงตัดสินใจ
# ทีละ flow เดี่ยวๆ แยกจากกันโดยสิ้นเชิง ทำให้แยกไม่ออกระหว่าง "เชื่อมต่อ
# ครั้งเดียวไปยัง port ที่ปิดอยู่" (เช่น nmap -sS ธรรมดา ส่ง SYN 1 ครั้ง
# แล้วโดน RST กลับทันที — duration สั้น, packet น้อย, dest_port อยู่ใน
# {21,22,...} พอดี ซึ่งมีรูปร่าง flow คล้าย 1 attempt ใน CICIDS
# SSH-Patator/FTP-Patator) กับ "พยายาม login ซ้ำๆ จริง" (เช่น hydra ที่
# ยิง connection จำนวนมากรัวๆ ในเวลาสั้นๆ) เลย ถ้าดูแค่ shape ของ flow
# เดี่ยวๆ ทีละอัน — นี่คือสาเหตุที่ nmap -sS ธรรมดา (ไม่มี login attempt
# ใดๆ) ทำให้เกิด BruteForce alert หลอกได้
#
# ทางแก้ที่ตรงประเด็นและไม่ต้อง retrain โมเดล: เพิ่ม "repetition gate"
# แบบ stateful หลัง ML ตัดสินว่าเป็น BruteForce candidate แล้ว — ต้องเห็น
# flow ที่ ML ทายว่าเป็น BruteForce ไปยัง (src_ip, dst_ip, dst_port)
# เดียวกัน อย่างน้อย BRUTEFORCE_MIN_ATTEMPTS ครั้ง ภายในหน้าต่างเวลา
# BRUTEFORCE_WINDOW_SECONDS วินาที ถึงจะยิง alert จริง — หลักการเดียวกับ
# ที่ signature-based IDS ใช้ตรวจ brute force มานาน (เช่น threshold-based
# rule ของ Suricata เอง หรือ fail2ban) คือเติม temporal aggregation ที่
# feature-level ML ยังขาดอยู่ ให้ตรงกับนิยามของคำว่า "brute force" เอง
# ไม่ใช่การกรองทิ้งหรือเดาจาก field ใดๆ
BRUTEFORCE_MIN_ATTEMPTS: int = 4
BRUTEFORCE_WINDOW_SECONDS: float = 20.0

# ─────────────────────────────────────────────
#  DoS — adaptive rate-anomaly detector (ไม่ hardcode)
# ─────────────────────────────────────────────
# DoS เชิงปริมาณ (HTTP flood เช่น GoldenEye/Hulk) เป็นปรากฏการณ์ "ระหว่าง
# flow" — per-flow ML มองไม่เห็นเพราะแต่ละ connection หน้าตาเหมือน GET ปกติ
# ต้องดูที่ "อัตรา" ข้าม flow
#
# แต่การตั้งเลขตายตัว (เช่น ">100 ครั้งใน 10 วิ") ผิดหลักเพราะ:
#   1) สมมติว่า DoS ต้อง "เร็ว" — DoS แบบส่งช้า ๆ (low-and-slow) จะหลุด
#   2) ต้อง "รอ" ให้ครบจำนวน = รอให้โจมตีสะสมก่อนถึงจับได้
#   3) เลขเดียวใช้ไม่ได้กับทุกเครือข่าย (บริการที่ปกติคนเยอะ vs เงียบ)
#
# วิธีที่ถูกหลักกว่า = adaptive statistical anomaly (แบบ sustained-rate):
# ระบบ "เรียนรู้ baseline" ของอัตราต่อต้นทางที่แต่ละบริการ (dst_ip,dst_port)
# เห็นตามปกติแบบออนไลน์ (EWMA) แล้ว flag ต้นทางที่มีอัตรา "เป็น outlier
# ทางสถิติ" (robust z-score เกิน DOS_ROBUST_Z) เทียบกับ baseline นั้น
#   - ปรับตามเครือข่ายเอง: บริการเงียบ baseline ต่ำ -> flood ช้า ๆ ก็เด่น
#     พอให้จับได้ / บริการคนเยอะ baseline สูง -> ไม่ false positive ง่าย
#   - เรียลไทม์: อัปเดตทุก event ไม่ต้องรอให้โจมตีจบ
#   - ไม่มีเลข "จำนวน" ตายตัว — เกณฑ์เดียวคือระดับนัยสำคัญทางสถิติ (z)
#     ซึ่งเป็นมาตรฐาน ไม่ใช่การเดา
#
# key ระดับ "อัตรา" ใช้ (src,dst,port) — flood ยิง port เดียวรัว -> อัตราสูง
# ส่วน port scan แตะหลาย port ละครั้ง -> อัตราต่อ (src,dst,port) ต่ำ ไม่ชนกัน
#
# *** 2 ปัญหาที่ต้องแก้ให้ถูกหลัก (v5.7) ***
# (ก) cold-start baseline poisoning: ถ้าบริการนั้นไม่มี traffic ปกติมาก่อน
#     เลย แล้วโดนโจมตีตั้งแต่แรก ระบบจะ "เรียน baseline จากตัวการโจมตี" เอง
#     (baseline สูงเท่า flood) จน z ไม่มีวันถึงเกณฑ์ = จับไม่ได้เลย
#     -> แก้ 2 ชั้น: (1) เริ่ม baseline ด้วย prior ต่ำ (สมมติบริการปกติเงียบ)
#        (2) อัปเดต baseline ด้วย "poisoning guard" ตั้งแต่ event แรก คือรับ
#        เฉพาะค่าที่อยู่ในเกณฑ์ปกติ (<= mean+3·mad) traffic โจมตีที่พุ่งสูง
#        จะไม่ถูกนำมาอัปเดต baseline -> baseline คงต่ำ -> z พุ่ง -> จับได้ทันที
#        (3) โหลด/บันทึก baseline ลงไฟล์ข้ามรอบ (persist) = ยิ่งรันนาน baseline
#        ยิ่งแม่น เหมือน anomaly-IDS มาตรฐานที่ profile normal ก่อน
# (ข) normal burst false positive: การเปิดเว็บ 1 หน้ายิง request หลายสิบใน
#     วินาทีเดียว ถ้าใช้หน้าต่างสั้นจะดูเหมือน flood -> ใช้หน้าต่าง "ยาวขึ้น"
#     (10s) เฉลี่ยออก burst สั้น ๆ ทิ้ง เหลือแต่ flood ที่ "รัวต่อเนื่อง" จริง
#     (GoldenEye ยิงต่อเนื่องหลายวินาที ผ่านการเฉลี่ยยังสูง / page-load spike
#     สั้น ๆ เฉลี่ยแล้วต่ำ) = แยก sustained flood ออกจาก burst ปกติได้
DOS_RATE_WINDOW_SECONDS: float = 10.0   # หน้าต่างวัดอัตรา (ยาวพอเฉลี่ย burst ปกติทิ้ง เหลือ sustained flood)
DOS_ROBUST_Z: float = 6.0               # flag เมื่ออัตราเกิน baseline นี้ (robust z)
DOS_MIN_RATE: float = 5.0               # noise gate: อัตราเฉลี่ยต่อ 10s ต้องเกินนี้ (req/s) ถึงพิจารณา
DOS_EWMA_ALPHA: float = 0.03            # ความเร็วปรับ baseline (ต่ำ = จำนาน กัน attack ดึง baseline)
DOS_PRIOR_RATE: float = 1.0             # baseline เริ่มต้นของบริการที่ยังไม่เคยเห็น (สมมติปกติเงียบ) กัน cold-start poisoning
DOS_BASELINE_FILE: str = "dos_baseline.json"  # persist baseline ข้ามรอบ (เก็บใน model_dir)
# *** สำคัญ: อัปเดต baseline ตาม "เวลา" ไม่ใช่ตาม "จำนวน event" ***
# ถ้าอัปเดตทุก event ตอนโดน flood ที่ยิง 700 req/s = อัปเดต baseline 700
# ครั้ง/วินาที baseline จะไล่ตามอัตราการโจมตีที่ค่อย ๆ ไต่ (ตอนหน้าต่างค่อย
# ๆ เต็ม) ได้ทัน จน z ไม่มีวันถึงเกณฑ์ — decouple ด้วยการอัปเดตอย่างมาก
# 1 ครั้งต่อ DOS_BASELINE_UPDATE_INTERVAL วินาที ต่อบริการ ทำให้ event
# volume ของ flood ดึง baseline ไม่ได้ (ช่วง flood baseline แทบไม่ขยับ)
DOS_BASELINE_UPDATE_INTERVAL: float = 5.0
# หลังจับ flood ได้ครั้งแรก ให้ถือว่า (src,dst,port) นั้น "ยังโจมตีอยู่" ต่อ
# อีก N วินาที แล้วลัดวงจร (ข้าม ML/payload) ทันที — ตัด false positive
# แบบ XSS/SQLi ที่เกิดจาก URL มั่ว ๆ ของเครื่องมือ flood (เช่น GoldenEye
# สุ่ม path เพื่อเลี่ยง cache) และลด CPU/backlog ระหว่างโดน flood ยาว ๆ
DOS_FLOOD_COOLDOWN_SECONDS: float = 15.0

# ─────────────────────────────────────────────
#  สวิตช์เปิด/ปิดชั้น "สถิติ/temporal" (ไม่ใช่ ML) — สำหรับทดสอบ/ablation
# ─────────────────────────────────────────────
# SWORD มี 3 ชั้นที่ "ไม่ใช่ ML" (นับเชิงเวลา/สถิติ) เสริมกับ ML flow/payload:
#   - DoS rate anomaly        (adaptive robust z-score)
#   - PortScan temporal fanout(นับ distinct port)
#   - BruteForce repetition   (ประตูนับซ้ำก่อนยิง — กัน false positive)
# ตั้ง False เพื่อปิดแต่ละตัวได้อิสระ (เช่นปิด DoS rate ตอนยิงด้วย automated
# tool อย่าง sqlmap ที่ยิงถี่จนไปเข้าเกณฑ์อัตรา หรือปิดหมดเพื่อดูผล ML ล้วน)
#
# *** ผลของการปิด (ต้องเข้าใจก่อนใช้) ***
#   ปิด ENABLE_DOS_RATE       -> per-flow ML จะยิง DoS เองแทน (ปกติโดน
#                                suppress ไว้เพราะไม่น่าเชื่อถือ — เป็น
#                                finding ในเล่ม: nmap probe ไป port เปิด
#                                หน้าตาคล้าย DoS flow ใน CICIDS จึงมักเป็น
#                                false positive) ปิดตัวนี้ = สลับจาก
#                                "adaptive rate เป็นตัวตัดสิน" ไปเป็น "ML
#                                ล้วนเป็นตัวตัดสิน" ไม่ใช่ปิด DoS ทั้งหมด
#                                — ไวขึ้นแต่ false positive เยอะขึ้นจริง
#                                ไม่แนะนำให้ปิดตอนใช้จริง เหมาะกับใช้เทียบผล
#                                ในเล่มว่าทำไมต้องมี adaptive rate layer
#   ปิด ENABLE_PORTSCAN_FANOUT -> "ไม่ตรวจ PortScan เลย" ด้วยเหตุผลเดียวกัน
#   ปิด ENABLE_BRUTEFORCE_GATE -> BruteForce ML จะยิงทันทีที่โมเดลทายว่าเป็น
#                                BruteForce โดยไม่ต้องรอทำซ้ำหลายครั้ง =
#                                ไวขึ้นแต่ false positive เยอะขึ้น (เช่น nmap
#                                connection เดียวอาจโดนแปะ BruteForce) —
#                                ไม่แนะนำให้ปิดตอนใช้จริง
ENABLE_DOS_RATE: bool = True
ENABLE_PORTSCAN_FANOUT: bool = True
ENABLE_BRUTEFORCE_GATE: bool = True

# ─────────────────────────────────────────────
#  Feature Selection
# ─────────────────────────────────────────────
# feature ทุกตัวในลิสต์นี้ต้อง "คำนวณได้จริงจาก Suricata eve.json ตอน
# inference สด" เท่านั้น — ห้ามมี feature ที่มีความหมายจริงแค่ตอนอ่านจาก
# CICIDS CSV แล้วกลายเป็นค่าคงที่ (fake) ตอนใช้งานจริง เพราะจะทำให้เกิด
# train/serve skew (โมเดลเรียนรู้ pattern จาก column ที่ live ไม่มีทาง
# reproduce ได้ แล้วพอ deploy จริงค่านั้นเป็น 0 เสมอ — พังแบบเงียบๆ ไม่มี
# error ให้เห็น) นี่คือหลักการที่ทำให้ตัดสินใจเอา tcp_syn/tcp_rst/tcp_ack
# ออกไป (ดูรายละเอียดด้านล่าง)
_BASE_FEATURES = [
    "dest_port", "duration", "total_fwd_packets", "total_bwd_packets",
    "total_packets", "flow_packets_per_sec",
    "duration_ms", "fwd_bwd_ratio", "pkt_ratio", "has_response",
    "flow_iat_mean", "is_long_connection", "log_duration",
    "pkts_per_duration", "acc_age", "n_flushes", "log_acc_age",
    "http_request_count", "http_method_count", "http_status_4xx_ratio",
    "http_status_5xx_ratio", "http_uri_len_avg", "http_uri_len_max",
    "http_param_count", "has_suspicious_chars",
]
# ── ตัดออกใน v5.1: feature กลุ่ม byte ──────────────────────────────
#   down_up_ratio, fwd_bytes_per_pkt, bwd_bytes_per_pkt,
#   bytes_ratio, pkt_size_ratio, flow_bytes_per_pkt
#
# เหตุผล (หลักฐานจาก check_cicids_flags.py บนไฟล์จริง):
#   CICIDS "Total Length of Fwd Packets" ของ PortScan มี median = 0
#   CICIDS "Fwd Packet Length Max"       ของ PortScan มี median = 0
#   แต่ Suricata รายงาน bytes_toserver = 60 สำหรับ SYN packet เดียวกัน
#
#   => CICFlowMeter นับ "เฉพาะ payload" (SYN ไม่มี payload จึงได้ 0)
#      Suricata นับ "รวม header" (SYN = 60 bytes)
#      เป็นคนละนิยาม และแปลงให้ตรงกันไม่ได้ เพราะเราไม่รู้ขนาด header
#      ของแต่ละแพ็กเก็ต (IPv4/IPv6 + TCP options ต่างกันไป)
#
# compare_train_live.py วัดผลกระทบจริงไว้:
#      fwd_bytes_per_pkt   เทรน=0     live=60    (ฝั่งหนึ่งเป็น 0)
#      flow_bytes_per_pkt  เทรน=3     live=57    (ต่าง 19 เท่า)
#      bytes_ratio         เทรน=0     live=1.09  (ฝั่งหนึ่งเป็น 0)
#
# down_up_ratio ตัดด้วยเหตุผลต่างออกไป: ฝั่งเทรนใช้คอลัมน์ "Down/Up Ratio"
# ของ CICFlowMeter ตรง ๆ ซึ่งเป็นอัตราส่วน "แพ็กเก็ต" แต่ฝั่ง live
# คำนวณเป็นอัตราส่วน "ไบต์" = คนละความหมาย แม้ค่าจะบังเอิญใกล้กันใน
# traffic ที่ทดสอบ (0.90x) ก็ไม่ควรพึ่ง — และ pkt_ratio ให้ความหมาย
# เดียวกันโดยผ่านการตรวจ parity แล้ว (1.00x)

# ── TCP flag features (กลับมาใน v4.0 แบบที่ parity ถูกต้อง) ────────
# ประวัติ: v3.6 ตัด tcp_syn/tcp_rst/tcp_ack ทิ้ง เพราะตอนนั้นโค้ดอ่านค่า
# จาก raw.get("tcp", {}) แล้วได้ {} ทำให้เป็น 0.0 คงที่ทุก flow ตอน live
# (train/serve skew แบบคลาสสิก — offline metric สวย แต่ live ตายสนิท)
# การตัดทิ้งตอนนั้น "ถูกต้อง" เมื่อเทียบกับการปล่อยค่าปลอมไว้ แต่ผลข้าง
# เคียงคือเสียลายเซ็นที่แยก PortScan ออกจาก BruteForce ไปด้วย เพราะความ
# ต่างระหว่างสองอย่างนี้อยู่ในสถานะ TCP ล้วน ๆ:
#     PortScan   : SYN -> RST     (ไม่เคย established)
#     BruteForce : SYN -> SYN/ACK -> ACK -> PSH (login จริง)
# เมื่อไม่มี flag เหลือให้ดู feature ที่เหลือมองสองอย่างนี้เหมือนกันหมด
# (duration สั้น, packet น้อย, ไม่มี payload) โมเดลจึงตกไปยึด dest_port
# เป็นตัวตัดสินหลัก แล้วเรียนกฎว่า "port 22 = BruteForce" (เพราะในชุด
# เทรน flow ไป port 21/22 เกือบทั้งหมดคือ FTP/SSH-Patator ส่วน PortScan
# กระจายทุก port) => nmap ยิง port 22 ก็เลยกลายเป็น BruteForce
#
# v4.0 เอากลับมาโดยแก้ที่ต้นเหตุจริง คือ "วิธีอ่านค่า" ไม่ใช่ตัด feature:
#   - ตอน live อ่านจาก raw["tcp"] ของ Suricata ซึ่งมีทั้ง boolean
#     (syn/rst/ack/fin/psh/urg) และ hex string (tcp_flags_ts/tcp_flags_tc)
#     โค้ดใหม่รองรับทั้งสองทาง และ parse hex เป็น fallback ให้ทนทานต่อ
#     ความต่างของ Suricata แต่ละเวอร์ชัน/คอนฟิก
#   - ตอนเทรนอ่านจากคอลัมน์ "SYN Flag Count" (2017) / "SYN Flag Cnt"
#     (2018) ของ CICFlowMeter
#   - *** binarize ทั้งสองฝั่ง *** (count > 0 -> 1.0) เพราะฝั่ง CICIDS
#     เป็น "จำนวนครั้ง" ส่วนฝั่ง Suricata เป็น "เคยเจอไหม" ถ้าไม่ binarize
#     สเกลจะคนละแบบ = train/serve skew รอบใหม่ทันที นี่คือจุดที่ต้องระวัง
#     ที่สุดของ patch นี้
TCP_FEATURES = [
    "tcp_syn", "tcp_rst", "tcp_ack", "tcp_fin", "tcp_psh",
]
# ── ตัดออกใน v4.1 หลังตรวจด้วย --verify-live บน eve.json จริง ────────
# (3,113 flow events จากเครือข่ายทดสอบจริง) พบว่า 3 ตัวนี้เป็นค่าคงที่
# 0 ตลอดตอน live = feature ตาย ถ้าปล่อยไว้จะเกิด train/serve skew ซ้ำรอย
# บั๊กเดิมที่เพิ่งแก้ไป:
#
#   tcp_urg         URG bit แทบไม่มีใครใช้จริงใน traffic ยุคนี้
#   tcp_syn_no_ack  } Suricata รวม flag "ทั้งสองทิศทาง" ไว้ใน object เดียว
#   tcp_rst_no_ack  } (สรุปทั้ง flow ไม่ใช่รายแพ็กเก็ต) — nmap -sS ยิง SYN
#                     ไป port ปิด ปลายทางตอบ RST ซึ่งตามมาตรฐาน TCP มี ACK
#                     ติดมาด้วยเสมอ (RST+ACK = 0x14) พอรวมสองทิศทาง flow
#                     ของ scan จึงมี ack=1 ทุกครั้ง เงื่อนไข "SYN แต่ไม่มี
#                     ACK" เลยเป็นเท็จเสมอ
#
# จะกู้ derived feature พวกนี้ต้องใช้ flag แยกทิศทาง (tcp_flags_ts /
# tcp_flags_tc ซึ่ง Suricata มีให้) แต่ทำไม่ได้ เพราะฝั่ง CICIDS ไม่มี
# SYN/RST/ACK แยกทิศทาง (มีแค่ Fwd/Bwd PSH กับ Fwd/Bwd URG เท่านั้น) —
# ถ้าใช้ทิศทางเฉพาะฝั่ง live ก็คือสร้าง skew รอบใหม่ จึงตัดทิ้งตาม
# หลักการเดิม: feature ไหนที่สองฝั่งคำนวณให้ตรงกันไม่ได้ ห้ามอยู่ใน FEATURES
#
# flag 5 ตัวที่เหลือแยก PortScan ออกจาก BruteForce ได้ด้วยตัวมันเองอยู่แล้ว:
#   PortScan   : syn=1 rst=1 ack=1 fin=0 psh=0   <- ไม่เคยมี data ไหลจริง
#   BruteForce : syn=1 rst=0 ack=1 fin=1 psh=1   <- มี login payload จริง
# คู่ (psh, fin) คือตัวแยกหลัก — scan ไม่มีทั้งคู่ session จริงมีทั้งคู่

# สวิตช์เดียวคุมทั้ง --train และ --realtime — ต้องเป็นค่าเดียวกันทั้งสอง
# ขั้นตอน ไม่งั้น feature vector ไม่ตรงกัน (โค้ดจะ "ปฏิเสธการรัน" ให้เอง
# ด้วย features.json compatibility check ใน _load_models() ไม่ปล่อยให้
# ทายมั่วเงียบ ๆ)
#
# ตั้งเป็น False เมื่อไหร่: ถ้ารัน `--verify-live` แล้วรายงานว่า TCP flag
# เป็นค่าคงที่ (Suricata ในเครื่องนั้นไม่ส่ง tcp object มากับ flow event)
# ระบบจะยังตรวจ PortScan ได้อยู่ผ่านชั้น temporal fan-out ด้านล่าง ซึ่ง
# ทำงานแยกจาก ML โดยสิ้นเชิง
USE_TCP_FEATURES: bool = False
# ── v5.1: ปิดถาวรตามหลักฐานจากข้อมูลจริง ───────────────────────────
# check_cicids_flags.py อ่านค่าดิบจาก CICIDS2017 PortScan file
# (286,467 แถว) — % ของแถวที่คอลัมน์นั้นมีค่า > 0 :
#
#                       PortScan    BENIGN
#       SYN Flag Count      0.0%      4.7%
#       RST Flag Count      0.0%      0.0%
#       ACK Flag Count      0.0%     27.8%
#       PSH Flag Count    100.0%     23.8%
#       FIN Flag Count      0.0%      2.0%
#
# ทุก TCP flow ต้องเริ่มด้วย SYN เสมอตาม RFC 793 แต่ข้อมูลบอกว่ามีแค่
# 4.7% ของ BENIGN และ 0% ของ PortScan ที่มี SYN — ส่วน PortScan ซึ่ง
# ไม่เคยส่ง application data กลับมี PSH ครบ 100% ค่าเหล่านี้ขัดกับ
# ความเป็นจริงของโปรโตคอล
#
# ยืนยันแล้วว่าไม่ใช่บั๊กการ map คอลัมน์ของเรา (ชื่อคอลัมน์ตรงทุกตัว)
# แต่เป็นข้อจำกัดของ CICFlowMeter ที่ใช้สร้าง dataset — มีงานวิจัยรายงาน
# ปัญหาคุณภาพของ CICIDS2017 ไว้หลายชิ้น เช่น Engelen et al. (2021)
# "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study"
#
# ผลกระทบที่วัดได้: โมเดลเรียนว่า "PortScan = ไม่มี SYN ไม่มี RST แต่มี
# PSH" ซึ่งตรงข้ามกับ scan จริงทุกข้อ (SYN=1 RST=1 PSH=0) จึงให้คะแนน
# 0.000 กับการสแกนจริง ทั้งที่ offline F1 = 0.9997 — เป็นตัวอย่างชัดเจน
# ว่า metric ตอน offline ไม่ได้รับประกันว่าใช้งานจริงได้ (Arp et al. P9)
#
# กลไกอ่าน TCP flag ฝั่ง live (_extract_tcp_flags) ยังอยู่ครบและทำงาน
# ถูกต้อง — ถ้าเปลี่ยนไปใช้ dataset ที่ flag เชื่อถือได้ ก็เปิดกลับเป็น
# True แล้ว retrain ได้ทันที

FEATURES = _BASE_FEATURES + (TCP_FEATURES if USE_TCP_FEATURES else [])

# ─────────────────────────────────────────────
#  Temporal fan-out (PortScan)
# ─────────────────────────────────────────────
# Port scan เป็นปรากฏการณ์ "ระหว่าง flow" โดยนิยาม — ข้อมูลที่บอกว่ามัน
# เป็น scan ("ต้นทางเดียวแตะไปหลาย port มาก") ไม่ได้อยู่ใน flow ใด flow
# หนึ่งเลย จึงไม่มีทางที่ per-flow classifier จะแทนมันได้ ไม่ว่าจะเลือก
# feature ดีแค่ไหน (nmap สแกน 1000 port = 1000 flow ที่แต่ละอันหน้าตา
# เหมือน "เชื่อมต่อสั้น ๆ ครั้งเดียว" ทั้งนั้น)
#
# ชั้นนี้จึงนับ "จำนวน port ไม่ซ้ำที่ src เดียวกันแตะ ภายในหน้าต่างเวลา"
# ตรง ๆ ซึ่งเป็นวิธีมาตรฐานที่ IDS ใช้จริงมานาน (Snort sfPortscan
# preprocessor, Suricata threshold rule) — ไม่ใช่การซ่อนข้อจำกัดของ ML
# แต่เป็นการยอมรับตามหลักการว่า per-flow ML แทนปรากฏการณ์นี้ไม่ได้ และ
# เป็น "ชั้นที่สอง" ที่ทำให้คำว่า Hybrid NIDS มีความหมายจริง
#
# นับเฉพาะ flow ที่ "ไม่มี payload จริง" (scan-like) เพื่อกัน false
# positive จาก client ที่เปิดหลาย connection จริง ๆ — เพราะ scan จะได้แค่
# RST หรือเงียบ ไม่มีการรับส่งข้อมูลจริง ส่วน traffic ปกติที่แตะหลาย port
# (เช่นเปิดเว็บ) จะมี payload กลับมาเสมอ
PORTSCAN_MIN_DISTINCT_PORTS: int = 15
PORTSCAN_WINDOW_SECONDS: float = 60.0
# flow ที่ถือว่า "scan-like": ฝั่งปลายทางแทบไม่ส่งอะไรกลับมาเลย
# (RST 1 แพ็กเก็ต หรือเงียบสนิท) = ไม่เคยมี session จริงเกิดขึ้น
PORTSCAN_SCANLIKE_MAX_BWD_BYTES: int = 100
PORTSCAN_SCANLIKE_MAX_BWD_PKTS: int = 2

CICIDS_COL_MAP = {
    # CICIDS2017 (verbose names)
    "Destination Port": "dest_port",
    "Flow Duration": "duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Down/Up Ratio": "down_up_ratio",
    "Flow IAT Mean": "flow_iat_mean",
    # CICIDS2018 (abbreviated CICFlowMeter names)
    "Dst Port": "dest_port",
    "Tot Fwd Pkts": "total_fwd_packets",
    "Tot Bwd Pkts": "total_bwd_packets",
    "TotLen Fwd Pkts": "total_fwd_bytes",
    "TotLen Bwd Pkts": "total_bwd_bytes",
    "Flow Byts/s": "flow_bytes_per_sec",
    "Flow Pkts/s": "flow_packets_per_sec",
    # TCP flag counts — CICIDS2017 (verbose)
    "SYN Flag Count": "_syn_cnt",
    "RST Flag Count": "_rst_cnt",
    "ACK Flag Count": "_ack_cnt",
    "FIN Flag Count": "_fin_cnt",
    "PSH Flag Count": "_psh_cnt",
    "URG Flag Count": "_urg_cnt",
    # TCP flag counts — CICIDS2018 (abbreviated)
    "SYN Flag Cnt": "_syn_cnt",
    "RST Flag Cnt": "_rst_cnt",
    "ACK Flag Cnt": "_ack_cnt",
    "FIN Flag Cnt": "_fin_cnt",
    "PSH Flag Cnt": "_psh_cnt",
    "URG Flag Cnt": "_urg_cnt",
    "Label": "Label",
}

# ชื่อกลาง (_xxx_cnt) -> ชื่อ feature หลัง binarize
_FLAG_CNT_TO_FEATURE = {
    "_syn_cnt": "tcp_syn", "_rst_cnt": "tcp_rst", "_ack_cnt": "tcp_ack",
    "_fin_cnt": "tcp_fin", "_psh_cnt": "tcp_psh", "_urg_cnt": "tcp_urg",
}

ML_LOG_DIR = "/var/log/sword_detection"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_nids.log")],
)
logger = logging.getLogger("hybrid-nids")

# ═══════════════════════════════════════════════════════
#  Signature-based XSS / SQLi detection
# ═══════════════════════════════════════════════════════

# แต่ละ pattern เป็น tuple: (keyword, attack_type)
# แยก list เพื่อให้ match แม่นยำขึ้น (ไม่ใช่แค่ "ใน uri มีคำนี้")
_XSS_PATTERNS = [
    ("<script", "Cross Site Scripting (XSS)"),
    ("alert(", "Cross Site Scripting (XSS)"),
    ("onerror=", "Cross Site Scripting (XSS)"),
    ("onload=", "Cross Site Scripting (XSS)"),
    ("javascript:", "Cross Site Scripting (XSS)"),
    ("onclick=", "Cross Site Scripting (XSS)"),
    ("onfocus=", "Cross Site Scripting (XSS)"),
    ("<svg", "Cross Site Scripting (XSS)"),
    ("prompt(", "Cross Site Scripting (XSS)"),
    ("document.cookie", "Cross Site Scripting (XSS)"),
]
_SQLI_PATTERNS = [
    ("' or ", "SQL Injection (SQLi)"),
    ("'or ", "SQL Injection (SQLi)"),
    ("1=1", "SQL Injection (SQLi)"),
    ("1=2", "SQL Injection (SQLi)"),
    ("union select", "SQL Injection (SQLi)"),
    ("union all select", "SQL Injection (SQLi)"),
    ("drop table", "SQL Injection (SQLi)"),
    ("select * from", "SQL Injection (SQLi)"),
    ("' ;", "SQL Injection (SQLi)"),
    ("';", "SQL Injection (SQLi)"),
    ("-- ", "SQL Injection (SQLi)"),
    ("/*", "SQL Injection (SQLi)"),
    ("xp_cmdshell", "SQL Injection (SQLi)"),
]


# ใช้แสดงผล specific_type จาก predicted_class ของ flow-ML (v3.5): เดิม
# WebAttack ที่ flow-model ทายจะถูกเดาเป็น "Brute Force Attack" เสมอ
# (เพราะ XSS/SQLi ถูกดักด้วย signature/payload-ML ไปก่อนหน้าแล้ว) ตอนนี้
# BruteForce เป็น OvR task แยกที่เทรนจริงแล้ว ไม่ต้องเดาอีกต่อไป — ถ้า
# flow-model ทาย WebAttack แปลว่าเจอ pattern ระดับ flow ของเว็บโจมตีที่
# ไม่ตรงกับ signature/payload ML เลย (เช่น XSS/SQLi ที่ signature จับไม่ทัน)
# จึงรายงานเป็น "unspecified subtype" ตรง ๆ แทนที่จะเดาผิดประเภท
_PREDICTED_CLASS_DISPLAY = {
    "DoS": "DoS / DDoS",
    "PortScan": "Port Scan",
    "WebAttack": "Web Attack (XSS/SQLi — unspecified subtype, ML flow-level)",
    "BruteForce": "Brute Force Attack",
    "Benign": "Benign",
}


def detect_web_signature(uri: str, status_code: str = "") -> tuple:
    """
    ตรวจสอบ XSS / SQLi จาก URI (signature-based)
    คืนค่า (specific_type, confidence) หรือ (None, 0.0) ถ้าไม่พบ

    ก่อนตรวจสอบจะ unquote URI ก่อน เพื่อป้องกัน encoded payload
    ( attacker encode '%27%20or%20'1'%3D'1 → decode เป็น ' or '1'='1 )
    """
    if not uri:
        return None, 0.0

    decoded = unquote(uri).lower()

    # SQLi ก่อน — มี pattern ที่ overlap กับ XSS บางตัว
    for keyword, attack_type in _SQLI_PATTERNS:
        if keyword in decoded:
            return attack_type, 1.0

    for keyword, attack_type in _XSS_PATTERNS:
        if keyword in decoded:
            return attack_type, 1.0

    # Brute-force indication: path ที่เป็นเป้าหมายแน่ๆ + 4xx
    if any(path in decoded for path in ["wp-login", "admin", "login.php",
                                         "xmlrpc.php", "wp-admin"]):
        try:
            if 400 <= int(status_code) < 500:
                return "Brute Force Attack", 0.8
        except (ValueError, TypeError):
            pass

    return None, 0.0


# ═══════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════


def _normalize_label_col(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col.lower() == "label":
            if col != "Label":
                df = df.rename(columns={col: "Label"})
            break
    return df


def load_and_preprocess(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.rename(columns=CICIDS_COL_MAP, inplace=True)

    label_found = False
    for col in df.columns:
        if col.lower() == "label":
            df.rename(columns={col: "Label"}, inplace=True)
            label_found = True
            break
    if not label_found:
        raise KeyError(
            f" ไม่พบคอลัมน์ 'Label' ในไฟล์ {os.path.basename(file_path)}"
        )

    # Coerce numeric columns (CICIDS2018 header rows leak into data)
    _NUMERIC = [
        "dest_port", "duration", "total_fwd_packets", "total_bwd_packets",
        "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_sec",
        "flow_packets_per_sec", "down_up_ratio", "flow_iat_mean",
    ] + list(_FLAG_CNT_TO_FEATURE.keys())
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["total_fwd_packets", "total_bwd_packets", "duration"]
    for r in required:
        if r not in df.columns:
            raise KeyError(f" ไม่พบคอลัมน์ '{r}'")

    # Fallback bytes columns
    for col in ["total_fwd_bytes", "total_bwd_bytes"]:
        if col not in df.columns:
            df[col] = 0

    # Feature engineering
    fwd_pkts = df["total_fwd_packets"]
    bwd_pkts = df["total_bwd_packets"]
    total_pkts = fwd_pkts + bwd_pkts
    fwd_safe = fwd_pkts.replace(0, 1)
    bwd_safe = bwd_pkts.replace(0, 1)
    fwd_bytes = df["total_fwd_bytes"].fillna(0)
    bwd_bytes = df["total_bwd_bytes"].fillna(0)

    df["total_packets"] = total_pkts
    df["duration_ms"] = df["duration"] / 1000.0
    df["flow_packets_per_sec"] = total_pkts / (df["duration"].replace(0, 1) / 1e6)
    df["fwd_bwd_ratio"] = fwd_pkts / (bwd_pkts + 1)
    df["pkt_ratio"] = bwd_pkts / fwd_safe
    df["has_response"] = (bwd_pkts > 0).astype(float)

    if "down_up_ratio" not in df.columns:
        df["down_up_ratio"] = bwd_bytes / fwd_safe
    df["fwd_bytes_per_pkt"] = fwd_bytes / fwd_safe
    df["bwd_bytes_per_pkt"] = bwd_bytes / bwd_safe
    df["bytes_ratio"] = df["fwd_bytes_per_pkt"] / (df["bwd_bytes_per_pkt"] + 1)
    df["pkt_size_ratio"] = df["fwd_bytes_per_pkt"] / (df["bwd_bytes_per_pkt"] + 1)
    df["flow_bytes_per_pkt"] = (fwd_bytes + bwd_bytes) / total_pkts.replace(0, 1)

    if "flow_iat_mean" not in df.columns:
        df["flow_iat_mean"] = 0.0

    # ── TCP flag features (binarize ให้ตรงกับฝั่ง Suricata) ──────────
    # CICFlowMeter ให้ "จำนวนครั้ง" ที่เจอ flag นั้นใน flow ส่วน Suricata
    # ให้แค่ "เคยเจอไหม" (boolean) — ต้อง binarize ฝั่งนี้ให้เป็น 0/1
    # เหมือนกัน ไม่งั้นสเกลคนละแบบ = train/serve skew รอบใหม่ทันที
    # (ดูคำอธิบายเต็มที่ comment เหนือ TCP_FEATURES ด้านบนไฟล์)
    for cnt_col, feat_col in _FLAG_CNT_TO_FEATURE.items():
        if cnt_col in df.columns:
            df[feat_col] = (pd.to_numeric(df[cnt_col], errors="coerce")
                            .fillna(0) > 0).astype(float)
        else:
            # ไฟล์นี้ไม่มีคอลัมน์ flag เลย — เติม 0 ไว้เพื่อให้ schema ครบ
            # (ถ้า USE_TCP_FEATURES=True แล้วเจอกรณีนี้บ่อย ๆ ควรตรวจสอบ
            #  ว่าใช้ dataset ถูกชุดหรือไม่ เพราะจะกลายเป็น feature ตายอีก)
            df[feat_col] = 0.0

    # หมายเหตุ v4.1: เลิกคำนวณ tcp_syn_no_ack / tcp_rst_no_ack แล้ว
    # เพราะพิสูจน์ด้วย --verify-live บนข้อมูลจริงว่าเป็น 0 ตลอดตอน live
    # (Suricata รวม flag สองทิศทาง -> RST ที่ตอบกลับมี ACK ติดมาเสมอ)
    # ดูคำอธิบายเต็มที่ comment ใต้ TCP_FEATURES ด้านบนไฟล์

    df["is_long_connection"] = (df["duration"] > 1_000_000).astype(float)
    df["log_duration"] = np.log10(df["duration"].clip(lower=1))
    df["pkts_per_duration"] = df["total_packets"] / df["log_duration"].replace(0, 1)

    dur_s = df["duration"] / 1e6
    df["acc_age"] = dur_s
    df["n_flushes"] = np.ceil(dur_s / 30.0).clip(lower=1)
    df["log_acc_age"] = np.log10(dur_s.clip(lower=1e-6) + 1)

    for col in ["http_request_count", "http_method_count", "http_status_4xx_ratio",
                "http_status_5xx_ratio", "http_uri_len_avg", "http_uri_len_max",
                "http_param_count", "has_suspicious_chars"]:
        if col not in df.columns:
            df[col] = 0.0

    df["Label"] = df["Label"].astype(str).str.strip().apply(_map_label)
    df.dropna(subset=["Label"], inplace=True)
    df = df[df["Label"].isin(TARGET_CLASSES)].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.drop_duplicates(inplace=True)
    return df


def balance_binary_arrays(X_train, y_train, ratio=2.0, random_state=42):
    rng = np.random.default_rng(random_state)
    idx_benign = np.where(y_train == 0)[0]
    idx_attack = np.where(y_train == 1)[0]
    n_benign = min(len(idx_benign), int(len(idx_attack) * ratio))
    chosen = rng.choice(idx_benign, size=n_benign, replace=False)
    idx = np.concatenate([chosen, idx_attack])
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


def _train_one_binary(X_tr, X_val, y_tr, y_val, label, param_grid):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score

    print(f"[{label}] Training set: {len(X_tr):,} samples  "
          f"(Attack={int(y_tr.sum()):,} / Benign={int((y_tr == 0).sum()):,})")

    # Decision Tree
    print(f"[{label}] Training Decision Tree...")
    dt_base = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=30, min_samples_split=60,
        random_state=42, class_weight="balanced",
    )
    dt = CalibratedClassifierCV(dt_base, method="sigmoid", cv=5).fit(X_tr, y_tr)
    dt_pred = dt.predict(X_val)
    dt_cv = cross_val_score(dt_base, X_tr, y_tr, cv=5, scoring="f1", n_jobs=-1)
    print(f"   [{label}] DT | CV F1={dt_cv.mean():.4f} ±{dt_cv.std():.4f} | Val F1 below")
    print(classification_report(y_val, dt_pred, target_names=["Benign", label],
                                zero_division=0, digits=4))

    # Random Forest
    print(f"[{label}] Training Random Forest...")
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10, max_features="sqrt",
        random_state=42, class_weight="balanced", n_jobs=-1,
    )
    rf = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5).fit(X_tr, y_tr)
    rf_pred = rf.predict(X_val)
    rf_cv = cross_val_score(rf_base, X_tr, y_tr, cv=5, scoring="f1", n_jobs=-1)
    print(f"[{label}] RF | CV F1={rf_cv.mean():.4f} ±{rf_cv.std():.4f} | Val F1 below")
    print(classification_report(y_val, rf_pred, target_names=["Benign", label],
                                zero_division=0, digits=4))

    # XGBoost + CalibratedClassifierCV
    print(f"[{label}] Training XGBoost (GridSearch cv=5)...")
    xgb_base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.5, min_child_weight=5,
    )
    grid = GridSearchCV(xgb_base, param_grid, cv=5, scoring="f1",
                        n_jobs=-1, verbose=0).fit(X_tr, y_tr)
    xgb_best = grid.best_estimator_
    xgb = CalibratedClassifierCV(xgb_best, method="sigmoid", cv=5).fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_val)
    print(f"   [{label}] XGB | best_params={grid.best_params_} | "
          f"CV F1={grid.best_score_:.4f} | Val F1 below")
    print(classification_report(y_val, xgb_pred, target_names=["Benign", label],
                                zero_division=0, digits=4))

    # CV vs Val gap
    from sklearn.metrics import f1_score
    vf1_dt = f1_score(y_val, dt_pred, zero_division=0)
    vf1_rf = f1_score(y_val, rf_pred, zero_division=0)
    vf1_xgb = f1_score(y_val, xgb_pred, zero_division=0)

    print(f"[{label}] CV vs Val gap:")
    for name, cv, vf in [("DT", dt_cv.mean(), vf1_dt),
                          ("RF", rf_cv.mean(), vf1_rf),
                          ("XGB", grid.best_score_, vf1_xgb)]:
        gap = abs(cv - vf)
        print(f"     {name:<3} → CV={cv:.4f}  Val={vf:.4f}  gap={gap:.4f}"
              + ("  overfit?" if gap > 0.01 else " ✅"))

    # F1-based weights
    f1s = [vf1_dt, vf1_rf, vf1_xgb]
    total = sum(f1s)
    if total > 0:
        w = [f / total for f in f1s]
    else:
        w = [DT_WEIGHT, RF_WEIGHT, XGB_WEIGHT]
    print(f"[{label}] F1-based weights → DT={w[0]:.3f}  RF={w[1]:.3f}  XGB={w[2]:.3f}")

    return (dt, rf, xgb), tuple(w)


def _auto_tune_threshold(dt, rf, xgb, X_val, y_val, label, weights=None):
    from sklearn.metrics import f1_score
    w_dt, w_rf, w_xgb = weights if weights else (DT_WEIGHT, RF_WEIGHT, XGB_WEIGHT)
    dt_p = dt.predict_proba(X_val)[:, 1]
    rf_p = rf.predict_proba(X_val)[:, 1]
    xgb_p = xgb.predict_proba(X_val)[:, 1]
    ensemble_p = w_dt * dt_p + w_rf * rf_p + w_xgb * xgb_p

    best_t, best_f1 = 0.5, 0.0
    rows = []
    for t in [round(x * 0.05, 2) for x in range(6, 19)]:
        f1 = f1_score(y_val, (ensemble_p >= t).astype(int), zero_division=0)
        rows.append((t, f1))
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    thr = "  ".join(f"t={t:.2f}→F1={f:.3f}{'★' if t == best_t else ''}"
                    for t, f in rows)
    print(f"[{label}] Threshold search:\n     {thr}")
    print(f"[{label}] Best threshold = {best_t:.2f}  (Ensemble F1 = {best_f1:.4f})")
    return best_t


def _auto_tune_threshold_xgb_only(xgb, X_val, y_val, label):
    """tune threshold สำหรับโหมด --xgb_only โดยเฉพาะ

    คะแนนของ XGB เดี่ยว ๆ อยู่คนละสเกลกับคะแนน ensemble ถ่วงน้ำหนัก ถ้า
    โหมด xgb_only ไปยืม threshold ของ ensemble มาใช้ ก็คือใช้ threshold
    ผิดสเกล ตัดสินใจเพี้ยนทั้งระบบ — จึง tune แยกไว้ตั้งแต่ตอนเทรน
    """
    from sklearn.metrics import f1_score
    p = xgb.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in [round(x * 0.05, 2) for x in range(6, 19)]:
        f1 = f1_score(y_val, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    print(f"[{label}] Best threshold (xgb_only) = {best_t:.2f}  (F1 = {best_f1:.4f})")
    return best_t


def train_models(dataset_path: str, model_dir: str = "./model"):
    signal.signal(signal.SIGINT, lambda s, f: (
        print("\nTraining interrupted by user (Ctrl+C)"),
        exit(0),
    ))

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ไฟล์จริงอยู่ใน subfolder ของ dataset_path เช่น
    # Dataset/CICIDS2017/MachineLearningCVE/... และ Dataset/CICIDS2018/CSV/...
    # (ตรวจจาก `find Dataset/CICIDS2017 -type f` / `find Dataset/CICIDS2018
    # -type f` ของเครื่องจริง) ไม่ใช่วางแบนราบใต้ dataset_path ตรง ๆ
    _C17 = "CICIDS2017/MachineLearningCVE"
    _C18 = "CICIDS2018/CSV"

    ATTACK_FILE_CANDIDATES = {
        "PortScan": [
            f"{_C17}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ],
        "DoS": [
            f"{_C17}/Wednesday-workingHours.pcap_ISCX.csv",
        ],
        "WebAttack": [
            f"{_C17}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            f"{_C18}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
            f"{_C18}/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
            f"{_C18}/02-22-2018.csv",
            f"{_C18}/02-23-2018.csv",
        ],
        # Brute Force — ไฟล์เดียวกับ WebAttack บางไฟล์ก็มี "Web Attack –
        # Brute Force" / "Brute Force -Web" ปนอยู่ (ถูกกรองแยกด้วย Label
        # ตอน filter ใน loop ด้านล่างอยู่แล้ว ไม่ทับซ้อนกับ WebAttack)
        # บวกไฟล์ Tuesday/Wednesday ที่มี FTP/SSH brute force ระดับ network
        "BruteForce": [
            f"{_C17}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # web login BF (2017)
            f"{_C18}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",       # web login BF (2018)
            f"{_C18}/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
            f"{_C17}/Tuesday-WorkingHours.pcap_ISCX.csv",                      # FTP/SSH-Patator (2017)
            f"{_C18}/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",      # FTP/SSH-BruteForce (2018)
        ],
    }

    param_grid = {
        "max_depth": [4, 6, 9],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [3, 5],
        "subsample": [0.7, 0.8],
    }
    attack_types = [("PortScan", "portscan"), ("DoS", "dos"), ("WebAttack", "webattack"),
                    ("BruteForce", "bruteforce")]
    tuned_thresholds = {}
    tuned_thresholds_xgb = {}   # สำหรับโหมด --xgb_only (คนละสเกลกับ ensemble)
    trained_weights = {}

    # ── v5.3: สร้าง shared benign pool จากทุกไฟล์ ──────────────────
    # รวม benign จากทุกไฟล์ที่มี เพื่อให้ทุก task เทรนกับ "ปกติ" ชุด
    # เดียวกัน (ดูเหตุผลเต็มที่ docstring ของ patch_v53) โหลด benign ทีละ
    # ไฟล์แล้วปล่อย df ทิ้ง เพื่อไม่ให้กิน RAM พร้อมกันทุกไฟล์
    _all_paths = []
    for _files in ATTACK_FILE_CANDIDATES.values():
        for _f in _files:
            _p = os.path.join(dataset_path, _f)
            if os.path.exists(_p) and _p not in _all_paths:
                _all_paths.append(_p)

    print(f"\n{'='*60}")
    print(f"สร้าง shared benign pool จาก {len(_all_paths)} ไฟล์")
    print(f"{'='*60}")
    _benign_parts = []
    for _p in _all_paths:
        try:
            _d = load_and_preprocess(_p)
        except Exception as e:
            print(f"  ข้าม {os.path.basename(_p)}: {e}")
            continue
        _b = _d[_d["Label"] == "Benign"]
        if len(_b) > PER_FILE_BENIGN_CAP:
            _b = _b.sample(n=PER_FILE_BENIGN_CAP, random_state=42)
        print(f"  {os.path.basename(_p):55s} benign {len(_b):,}")
        _benign_parts.append(_b.copy())
        del _d, _b
    shared_benign = (pd.concat(_benign_parts, ignore_index=True)
                     if _benign_parts else pd.DataFrame(columns=FEATURES + ["Label"]))
    shared_benign = shared_benign.drop_duplicates().reset_index(drop=True)
    print(f"รวม shared benign pool: {len(shared_benign):,} rows "
          f"(ใช้เป็น negative ของทุก task)")

    for attack_label, attack_dir in attack_types:
        candidates = ATTACK_FILE_CANDIDATES.get(attack_label, [])
        found = [f for f in candidates
                 if os.path.exists(os.path.join(dataset_path, f))]
        if not found:
            print(f"Not found any of {candidates} — skip {attack_label}")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {attack_label}  ←  {found}")
        print(f"{'='*60}")
        t0 = time.time()

        # attack rows จากไฟล์ของ task นี้ (โหลดซ้ำ — benign จาก pool แทน)
        parts = [load_and_preprocess(os.path.join(dataset_path, f))
                 for f in found]
        task_df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        attack_df = task_df[task_df["Label"] == attack_label]
        # v5.3: negative class = shared benign pool (เหมือนกันทุก task)
        # แทนที่จะเป็น benign เฉพาะของไฟล์ task นี้
        df = pd.concat([attack_df, shared_benign], ignore_index=True)
        df = df[df["Label"].isin(["Benign", attack_label])].copy()
        del task_df, attack_df, parts

        vc = df["Label"].value_counts()
        print(f"Dataset after filter: {len(df):,} rows")
        for cls, cnt in vc.items():
            print(f"     {cls:<12}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")

        # ── ตรวจ dead feature ฝั่งข้อมูลเทรน ───────────────────────
        # คู่ตรงข้ามของ --verify-live (ซึ่งตรวจฝั่ง live) — feature ที่เป็น
        # ค่าคงที่ในชุดเทรนคือ feature ที่โมเดลเรียนรู้อะไรจากมันไม่ได้เลย
        # และถ้ามันไม่คงที่ตอน live ก็คือ train/serve skew อีกทิศหนึ่ง
        # เตือนตั้งแต่ตอนเทรน ดีกว่าไปเจอตอน deploy เหมือนที่ผ่านมา
        _const = [c for c in FEATURES if df[c].nunique(dropna=True) <= 1]
        if _const:
            print(f"[{attack_label}] ⚠️  feature ที่เป็นค่าคงที่ในข้อมูลเทรน "
                  f"{len(_const)} ตัว — โมเดลเรียนรู้จากมันไม่ได้:")
            for c in _const:
                print(f"       - {c} (= {df[c].iloc[0]})")
            print(f"[{attack_label}]    ถ้าตัวเดียวกันนี้ 'ไม่คงที่' ตอน live "
                  f"(เช็คด้วย --verify-live) แปลว่ามี skew ต้องแก้ก่อนใช้จริง")
        else:
            print(f"[{attack_label}] ✅ ทุก feature มีความแปรผันในข้อมูลเทรน")

        X_raw = df[FEATURES].values
        y_raw = (df["Label"] == attack_label).astype(int).values

        # Train / Val / Test = 60 / 20 / 20
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_raw, y_raw, test_size=0.4, random_state=42, stratify=y_raw)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

        print(f"Split → Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        X_tr_b, y_tr_b = balance_binary_arrays(X_tr_s, y_tr)
        print(f"After balance → {len(X_tr_b):,} "
              f"(Attack={int(y_tr_b.sum()):,} / Benign={int((y_tr_b == 0).sum()):,})")

        models, weights = _train_one_binary(
            X_tr_b, X_val_s, y_tr_b, y_val, attack_label, param_grid)
        dt, rf, xgb = models
        trained_weights[attack_label] = weights

        sub_dir = os.path.join(model_dir, attack_dir)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)
        for fn, obj in [("dt_model.pkl", dt), ("rf_model.pkl", rf),
                        ("xgb_model.pkl", xgb), ("scaler.pkl", scaler)]:
            with open(os.path.join(sub_dir, fn), "wb") as f:
                pickle.dump(obj, f)

        thr = _auto_tune_threshold(dt, rf, xgb, X_val_s, y_val,
                                   attack_label, weights)
        tuned_thresholds[attack_label] = thr
        tuned_thresholds_xgb[attack_label] = _auto_tune_threshold_xgb_only(
            xgb, X_val_s, y_val, attack_label)

        # Honest test report (ไม่เคยใช้ tune weight/threshold/max_depth/etc)
        w_dt, w_rf, w_xgb = weights
        test_p = (w_dt * dt.predict_proba(X_te_s)[:, 1]
                  + w_rf * rf.predict_proba(X_te_s)[:, 1]
                  + w_xgb * xgb.predict_proba(X_te_s)[:, 1])
        test_pred = (test_p >= thr).astype(int)
        print(f"\n[{attack_label}] ══ Held-out TEST report (not used in tuning) ══")
        print(classification_report(y_te, test_pred,
                                    target_names=["Benign", attack_label],
                                    zero_division=0, digits=4))
        print(f"Models saved → {sub_dir}")
        print(f"{attack_label} total time: {time.time() - t0:.1f}s")

    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump(FEATURES, f)
    with open(os.path.join(model_dir, "thresholds.json"), "w") as f:
        json.dump(tuned_thresholds, f, indent=2)
    with open(os.path.join(model_dir, "thresholds_xgb.json"), "w") as f:
        json.dump(tuned_thresholds_xgb, f, indent=2)
    with open(os.path.join(model_dir, "weights.json"), "w") as f:
        json.dump(trained_weights, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete | Thresholds: {tuned_thresholds}")
    print(f"Weights: {trained_weights}")
    print(f"{'='*60}\n")
    return [a for a, _ in attack_types]


# ═══════════════════════════════════════════════════════
#  HYBRID NIDS — Inference
# ═══════════════════════════════════════════════════════

class HybridNIDS:
    def __init__(self, model_dir="./model", use_telegram=False,
                 threshold=DEFAULT_THRESHOLD, xgb_only=False):
        self.model_dir = model_dir
        self.use_telegram = use_telegram
        self.threshold = threshold
        # โหมดทดสอบเร็ว: ใช้ XGBoost model เดียวต่อ task แทน DT+RF+XGB
        # ensemble ทั้งหมด — ไม่ retrain ไฟล์ .pkl เดิมยังใช้ได้ปกติ แค่
        # ข้ามการเรียก dt/rf ตอน inference เท่านั้น ดู comment เต็ม ๆ ที่
        # patch_xgb_only_mode.py
        self.xgb_only = xgb_only
        if self.xgb_only:
            logger.info("โหมด --xgb_only: ใช้เฉพาะ XGBoost ต่อ task "
                        "(ข้าม DT/RF) — เร็วขึ้นแต่ threshold ที่ tune ไว้ "
                        "อาจไม่ optimal เป๊ะ ใช้ทดสอบชั่วคราวเท่านั้น")
        self._load_models()
        self.alert_count = 0
        self._alert_cache = {}
        self.local_ips = self._get_local_ips()
        # ป้องกัน race condition ตอน monitor_realtime() ประมวลผลหลาย event
        # พร้อมกันจากหลาย worker thread — handle_alert() แก้ shared state
        # (_alert_cache, alert_count) และเขียนไฟล์ log ร่วมกัน ต้อง lock
        self._alert_lock = threading.Lock()
        # BruteForce repetition gate state — ดู comment เหนือ
        # BRUTEFORCE_MIN_ATTEMPTS ด้านบนของไฟล์
        self._bf_lock = threading.Lock()
        self._bf_attempts = defaultdict(deque)
        # PortScan temporal fan-out state — ดู comment เหนือ
        # PORTSCAN_MIN_DISTINCT_PORTS ด้านบนของไฟล์
        # โครงสร้าง: {(src_ip, dst_ip): deque[(timestamp, dest_port)]}
        self._ps_lock = threading.Lock()
        self._ps_events = defaultdict(deque)
        self._ps_last_alert = {}
        # DoS adaptive rate-anomaly — ดู comment เหนือ DOS_ROBUST_Z
        # _dos_win  : {(src,dst,port): deque[ts]} วัดอัตราปัจจุบันของต้นทาง
        # _svc_stats: {(dst,port): {mean,mad,n}} baseline อัตราต่อต้นทางของบริการ
        self._dos_lock = threading.Lock()
        self._dos_win = defaultdict(deque)
        # baseline เริ่มด้วย prior ต่ำ (สมมติบริการปกติเงียบ) กัน cold-start
        # poisoning — ดู comment เหนือ DOS_PRIOR_RATE
        self._svc_stats = defaultdict(
            lambda: {"mean": DOS_PRIOR_RATE, "mad": DOS_PRIOR_RATE,
                     "n": 0, "last_update": 0.0})
        self._flood_last = {}   # (src,dst,port) -> event-time ที่จับ flood ล่าสุด (cooldown)
        self._load_dos_baseline()   # โหลด baseline ที่ profile ไว้จากรอบก่อน (ถ้ามี)

    def _get_local_ips(self):
        import socket
        ips = {"127.0.0.1", "::1"}
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                ips.add(info[4][0].split("%")[0])
        except Exception:
            pass
        logger.info(f"Local IPs (will be ignored): {ips}")
        return ips

    # OvR flow tasks ที่ใช้ตัดสินใจ alert จริง — "webattack" (flow-level
    # XSS/SQLi) ถูกตัดออกโดยตั้งใจ (v3.6): แม้เทรนสำเร็จและแก้บั๊ก encoding
    # ที่ทำให้ข้อมูลหายไปแล้ว (317 -> 990 rows) แต่ precision บน held-out
    # test ยังต่ำมาก (0.083 = false positive ~11 ครั้งต่อ true positive 1
    # ครั้ง) เพราะ XSS/SQLi ที่ระดับ flow หายากเกินไปใน CICIDS ทั้งตระกูล
    # (0.044% ของข้อมูล) ไม่พอสำหรับ binary classifier ที่เชื่อถือได้จริง
    # ตัวจับ XSS/SQLi หลักตอนนี้คือ payload-ML pipeline (TF-IDF+XGBoost,
    # ยัง ML เหมือนเดิม แม่นกว่ามาก F1 0.98-0.99 ทั้ง in-distribution และ
    # OOD) + signature override ใน _analyze_http_event() — สคริปต์เทรน
    # (train_models()) ยังเทรน+เซฟโมเดล webattack ไว้ตามเดิมเผื่อใช้อ้างอิง/
    # เทียบผลในเล่มจบ แค่ตอน inference ไม่โหลดเข้ามาตัดสินใจ alert เท่านั้น
    _RUNTIME_FLOW_TASKS = ["portscan", "dos", "bruteforce"]

    def _load_models(self):
        md = self.model_dir
        self.features = json.load(open(os.path.join(md, "features.json")))
        # ── Compatibility check ────────────────────────────────────
        # บั๊กที่กัดกินโปรเจกต์นี้มาตลอดคือ "โมเดลถูกเทรนด้วย feature ชุด
        # หนึ่ง แต่ตอน inference โค้ดสร้าง feature อีกชุดหนึ่ง" แล้วไม่มี
        # error อะไรเลย ได้แต่ทายมั่วเงียบ ๆ (predict() ใช้ reindex ซึ่ง
        # เติม 0 ให้คอลัมน์ที่ขาดโดยไม่บ่น) — ตรงนี้จึงเช็คตรง ๆ แล้ว
        # "ปฏิเสธการรัน" ถ้าไม่ตรง ดีกว่าปล่อยให้ผลลัพธ์ผิดแบบเงียบ ๆ
        if list(self.features) != list(FEATURES):
            only_model = [f for f in self.features if f not in FEATURES]
            only_code = [f for f in FEATURES if f not in self.features]
            raise SystemExit(
                "\n" + "=" * 68 + "\n"
                "[FATAL] feature set ของโมเดลไม่ตรงกับโค้ดปัจจุบัน\n"
                + "=" * 68 + "\n"
                f"  โมเดลใน {md} เทรนด้วย {len(self.features)} features\n"
                f"  โค้ดปัจจุบันสร้าง        {len(FEATURES)} features\n"
                f"  มีเฉพาะในโมเดล (ต้อง retrain): {only_model or '-'}\n"
                f"  มีเฉพาะในโค้ด  (ต้อง retrain): {only_code or '-'}\n\n"
                "  วิธีแก้: เทรนใหม่ด้วยโค้ดเวอร์ชันนี้\n"
                f"      python3 hybrid_nids.py --train <DATASET> --model_dir {md}\n"
                "  (ถ้าตั้งใจใช้โมเดลเก่า ต้องตั้ง USE_TCP_FEATURES ให้ตรงกับ\n"
                "   ตอนที่เทรนโมเดลนั้น แล้วรันใหม่)\n"
                + "=" * 68
            )
        self.models_ovr = {}
        for att in self._RUNTIME_FLOW_TASKS:
            sd = os.path.join(md, att)
            if os.path.exists(sd):
                lbl = {"portscan": "PortScan", "dos": "DoS",
                       "webattack": "WebAttack", "bruteforce": "BruteForce"}[att]
                entry = {
                    "xgb": pickle.load(open(os.path.join(sd, "xgb_model.pkl"), "rb")),
                    "scaler": pickle.load(open(os.path.join(sd, "scaler.pkl"), "rb")),
                    "label": lbl,
                }
                if not self.xgb_only:
                    entry["dt"] = pickle.load(open(os.path.join(sd, "dt_model.pkl"), "rb"))
                    entry["rf"] = pickle.load(open(os.path.join(sd, "rf_model.pkl"), "rb"))
                self.models_ovr[att] = entry
        # โหมด xgb_only ต้องใช้ threshold ที่ tune จากคะแนน XGB เดี่ยว ๆ
        # ไม่ใช่ของ ensemble (คนละสเกลกัน) — ดู _auto_tune_threshold_xgb_only
        thr_path = os.path.join(md, "thresholds.json")
        if self.xgb_only:
            xgb_thr_path = os.path.join(md, "thresholds_xgb.json")
            if os.path.exists(xgb_thr_path):
                thr_path = xgb_thr_path
                logger.info("ใช้ thresholds_xgb.json (tune สำหรับ XGB เดี่ยว)")
            else:
                logger.warning(
                    "โหมด --xgb_only แต่ไม่พบ thresholds_xgb.json "
                    "— ใช้ threshold ของ ensemble ไปก่อน ซึ่งเป็นคนละสเกล "
                    "อาจตัดสินใจเพี้ยน ควรเทรนใหม่ด้วยโค้ดเวอร์ชันนี้")
        self.tuned_thresholds = (
            json.load(open(thr_path)) if os.path.exists(thr_path) else {}
        )
        w_path = os.path.join(md, "weights.json")
        self.ensemble_weights = (
            json.load(open(w_path)) if os.path.exists(w_path) else {}
        )
        # ── Payload ML (TF-IDF + XGBoost) — Pipeline 2 ──────────
        self.payload_vec = None
        self.payload_xgb = None
        self.payload_threshold = 0.85  # fallback เผื่อไม่มี payload_meta.json (โมเดลเก่า)
        try:
            self.payload_vec = pickle.load(
                open(os.path.join(md, "payload_vectorizer.pkl"), "rb"))
            self.payload_xgb = pickle.load(
                open(os.path.join(md, "payload_xgb.pkl"), "rb"))
            meta_path = os.path.join(md, "payload_meta.json")
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                self.payload_threshold = float(meta.get("threshold", 0.85))
                logger.info(f"Payload ML loaded (XSS/SQLi) — tuned threshold={self.payload_threshold:.2f}")
            else:
                logger.info("Payload ML loaded (XSS/SQLi) — ไม่พบ payload_meta.json "
                            "ใช้ threshold default 0.85 (โมเดลนี้เทรนด้วย train_payload.py "
                            "เวอร์ชันเก่าที่ยังไม่ tune threshold จาก validation set)")
        except Exception as e:
            # บอก path ที่ไปหาจริง ๆ + สิ่งที่ขาด แทนที่จะบอกลอย ๆ ว่า
            # "ไม่พบ" เพราะต้นตอที่เจอบ่อยที่สุดคือเทรน payload ไว้คนละ
            # model_dir กับที่ --realtime ใช้ ซึ่งดูจากข้อความเดิมไม่ออกเลย
            want = ["payload_vectorizer.pkl", "payload_xgb.pkl", "payload_meta.json"]
            missing = [n for n in want
                       if not os.path.exists(os.path.join(md, n))]
            logger.warning(
                "ไม่พบ Payload ML (XSS/SQLi) — ทำงานแบบ signature-only\n"
                f"    model_dir ที่ใช้อยู่ : {os.path.abspath(md)}\n"
                f"    ไฟล์ที่ยังขาด        : {missing if missing else 'ครบ แต่โหลดไม่ได้'}\n"
                f"    สาเหตุที่โหลดไม่สำเร็จ: {type(e).__name__}: {e}\n"
                "    วิธีแก้: เทรน payload ลง model_dir เดียวกันนี้\n"
                f"        python3 train_payload.py web_payloads_merged.csv --model_dir {md}\n"
                "    (ระวัง: ถ้าเทรนไว้คนละโฟลเดอร์/คนละเครื่อง จะไม่เจอ)"
            )

    def predict(self, features: pd.DataFrame) -> dict:
        try:
            X_df = features.reindex(columns=self.features, fill_value=0)
            scores = {}
            per_model = {}
            for att, m in self.models_ovr.items():
                label = m["label"]
                Xs = m["scaler"].transform(X_df.values)
                xgb_p = float(m["xgb"].predict_proba(Xs)[0][1])
                if self.xgb_only:
                    scores[label] = xgb_p
                    per_model[label] = {"xgb": xgb_p}
                else:
                    dt_p = float(m["dt"].predict_proba(Xs)[0][1])
                    rf_p = float(m["rf"].predict_proba(Xs)[0][1])
                    w = self.ensemble_weights.get(
                        label, (DT_WEIGHT, RF_WEIGHT, XGB_WEIGHT))
                    scores[label] = w[0] * dt_p + w[1] * rf_p + w[2] * xgb_p
                    per_model[label] = {"dt": dt_p, "rf": rf_p, "xgb": xgb_p}

            # ใช้ raw ensemble score ตรงๆ (scale เดียวกับตอน tune threshold)
            max_s = max(scores.values())
            best_l = max(scores, key=scores.get)
            eff_t = self.tuned_thresholds.get(best_l, self.threshold)

            sorted_s = sorted(scores.values(), reverse=True)
            second = sorted_s[1] if len(sorted_s) > 1 else 0.0
            gap_ok = (max_s - second) >= SECONDARY_REJECTION_GAP

            is_atk = (max_s >= eff_t) and gap_ok

            return {
                "predicted_class": best_l if is_atk else "Benign",
                "confidence": max_s if is_atk else 1.0 - max_s,
                "is_attack": is_atk,
                "model_conf": per_model[best_l],
                "all_scores": scores,
                "all_per_model": per_model,
            }
        except Exception as e:
            logger.warning(f"predict() exception: {e}")
            return {"predicted_class": "Unknown", "is_attack": False}

    def _build_feature_row(self, raw, http_data=None):
        """
        สร้าง feature dict จาก Suricata raw event
        รองรับทั้ง flow event และ http event (http_data dict)
        """
        flow = raw.get("flow", {}) or {}
        # duration ต้องละเอียดระดับไมโครวินาทีให้ตรงกับ CICFlowMeter ตอน
        # เทรน — flow.age ของ Suricata เป็นจำนวนเต็ม "วินาที" ใช้ไม่ได้
        # (ดูคำอธิบายเต็มที่ _flow_duration_seconds)
        dur_s = self._flow_duration_seconds(flow)
        spkts = int(flow.get("pkts_toserver", 0))
        dpkts = int(flow.get("pkts_toclient", 0))
        sbytes = int(flow.get("bytes_toserver", 0))
        dbytes = int(flow.get("bytes_toclient", 0))
        dst_port = int(raw.get("dest_port", 0) or 0)
        total = spkts + dpkts

        # ── TCP flags จาก Suricata (v4.0) ───────────────────────
        # รองรับ 2 รูปแบบที่ Suricata ส่งมา (ต่างกันตามเวอร์ชัน/คอนฟิก):
        #   (ก) boolean ตรง ๆ : {"syn": true, "ack": false, ...}
        #   (ข) hex string     : {"tcp_flags_ts": "02", "tcp_flags_tc": "14"}
        # ใช้ (ก) ก่อน ถ้าไม่มีค่อย parse (ข) — ทนทานกว่าอ่านทางเดียว
        # ผลลัพธ์ binarize เป็น 0.0/1.0 ให้ตรงกับฝั่งเทรนที่ binarize จาก
        # flag count เหมือนกัน (ดู comment เหนือ TCP_FEATURES)
        tcp_flags = self._extract_tcp_flags(raw)

        dur_us = dur_s * 1e6
        log_dur = np.log10(max(dur_us, 1))
        fwd_safe = max(spkts, 1)
        bwd_safe = max(dpkts, 1)
        total_bytes = sbytes + dbytes
        fwd_bpp = sbytes / fwd_safe
        bwd_bpp = dbytes / bwd_safe
        app_proto = raw.get("app_proto", "")

        # HTTP features
        http_req = 1 if http_data or app_proto == "http" else 0
        http_mth = 1 if http_data and http_data.get("http_method") else 0
        status_4xx = 0.0
        status_5xx = 0.0
        uri_len_avg = 0.0
        uri_len_max = 0.0
        param_cnt = 0
        suspicious = 0

        if http_data:
            uri = http_data.get("url", "")
            status = str(http_data.get("status", "200"))
            try:
                si = int(status)
                status_4xx = 1.0 if 400 <= si < 500 else 0.0
                status_5xx = 1.0 if 500 <= si < 600 else 0.0
            except (ValueError, TypeError):
                pass
            uri_len_avg = float(len(uri))
            uri_len_max = float(len(uri))
            if uri and "?" in uri:
                param_cnt = uri.split("?", 1)[1].count("&") + 1
            _sig_type, _ = detect_web_signature(uri, status)
            suspicious = 1 if _sig_type else 0

        return {
            "dest_port": dst_port,
            "duration": dur_us,
            "duration_ms": dur_s * 1000,
            "total_fwd_packets": spkts,
            "total_bwd_packets": dpkts,
            "total_packets": total,
            "flow_packets_per_sec": total / max(dur_s, 1e-6),
            "down_up_ratio": dbytes / max(sbytes, 1),
            "fwd_bytes_per_pkt": fwd_bpp,
            "bwd_bytes_per_pkt": bwd_bpp,
            "bytes_ratio": fwd_bpp / (bwd_bpp + 1),
            "pkt_size_ratio": fwd_bpp / (bwd_bpp + 1),
            "flow_bytes_per_pkt": total_bytes / max(total, 1),
            "fwd_bwd_ratio": spkts / (dpkts + 1),
            "pkt_ratio": dpkts / max(spkts, 1),
            "has_response": float(dpkts > 0),
            "flow_iat_mean": dur_us / max(total - 1, 1),
            "is_long_connection": float(dur_s > 1),
            "log_duration": log_dur,
            "pkts_per_duration": total / max(log_dur, 1),
            "acc_age": dur_s,
            "n_flushes": np.ceil(dur_s / 30),
            "log_acc_age": np.log10(dur_s + 1),
            "http_request_count": http_req,
            "http_method_count": http_mth,
            "http_status_4xx_ratio": status_4xx,
            "http_status_5xx_ratio": status_5xx,
            "http_uri_len_avg": uri_len_avg,
            "http_uri_len_max": uri_len_max,
            "http_param_count": param_cnt,
            "has_suspicious_chars": float(suspicious),
            # TCP flags — คีย์เหล่านี้ถูกใส่มาเสมอ ส่วนจะถูกใช้จริงหรือไม่
            # ขึ้นกับ USE_TCP_FEATURES ผ่าน FEATURES (predict() ใช้
            # reindex(columns=self.features) อยู่แล้ว คีย์เกินจะถูกตัดทิ้ง
            # เองอย่างปลอดภัย)
            **tcp_flags,
        }

    # นับว่าต้อง fallback ไปใช้ flow.age กี่ครั้ง (ใช้เตือนตอนจบ)
    _dur_fallback = 0
    _dur_total = 0

    @classmethod
    def _flow_duration_seconds(cls, flow: dict) -> float:
        """คืน duration ของ flow เป็นวินาที (ทศนิยม ละเอียดไมโครวินาที)

        ทำไมไม่ใช้ flow.age ตรง ๆ: Suricata ส่ง age มาเป็น "จำนวนเต็ม
        วินาที" — พิสูจน์แล้วด้วย --verify-live บนข้อมูลจริง (3,113 flow
        มี acc_age แค่ 7 ค่าไม่ซ้ำ: 0..31) ในขณะที่ฝั่งเทรน คอลัมน์
        "Flow Duration" ของ CICFlowMeter เป็นไมโครวินาที ละเอียด 1e-6

        ผลคือ flow สั้น ๆ ทุกอันตอน live ได้ age=0 -> duration=0 ซึ่งก็คือ
        *ทุก flow ของ port scan* (SYN->RST จบใน ~100 ไมโครวินาที) แต่ตอน
        เทรน flow แบบเดียวกันมี duration 100-5000 ไมโครวินาที = train/serve
        skew ที่ทำให้ feature 10 ตัวจาก 36 เพี้ยนพร้อมกัน และเป็นสาเหตุ
        โดยตรงที่ PortScan ML ให้คะแนน 0.000 กับ scan จริง ทั้งที่ offline
        F1 = 0.9997

        flow.start / flow.end เป็น ISO8601 ที่มีไมโครวินาทีอยู่แล้ว
        (เช่น "2026-08-23T16:32:11.123456+0700") จึงคำนวณจากสองตัวนี้แทน
        ได้ความละเอียดตรงกับฝั่งเทรนพอดี
        """
        start, end = flow.get("start"), flow.get("end")
        # นับสถิติเฉพาะ "flow event จริง" (flow dict ไม่ว่าง) เท่านั้น —
        # HTTP event ก็เรียกฟังก์ชันนี้แต่ raw["flow"] มักว่าง ({}) และ
        # duration ไม่สำคัญกับ payload ML อยู่แล้ว ถ้านับรวมด้วยจะทำให้ %
        # fallback ดูสูงหลอกตา ทั้งที่ไม่กระทบการตรวจจับเลย
        is_real_flow = bool(flow)
        if is_real_flow:
            cls._dur_total += 1
        if start and end:
            try:
                d = (datetime.fromisoformat(end)
                     - datetime.fromisoformat(start)).total_seconds()
                if d >= 0:
                    return d
            except (ValueError, TypeError):
                pass
        # fallback: ไม่มี start/end หรือ parse ไม่ได้ -> ใช้ age (หยาบ)
        if is_real_flow:
            cls._dur_fallback += 1
        return float(flow.get("age", 0) or 0)

    @staticmethod
    def _extract_tcp_flags(raw: dict) -> dict:
        """ดึง TCP flag จาก Suricata flow event -> dict ของ 0.0/1.0

        อ่านได้ 2 ทาง (ดู comment ใน _build_feature_row): boolean field
        ตรง ๆ ก่อน ถ้าไม่มีค่อย parse hex string tcp_flags_ts/tc
        คืน 0.0 ทั้งหมดถ้าไม่มี tcp object เลย (เช่น UDP/ICMP flow ซึ่ง
        ถูกต้องตามความหมาย — ไม่ใช่ค่าปลอม เพราะ flow พวกนั้นไม่มี TCP
        flag อยู่จริง ๆ)
        """
        tcp = raw.get("tcp") or {}
        names = ("fin", "syn", "rst", "psh", "ack", "urg")
        # bit position ตามมาตรฐาน TCP header: FIN=0x01 SYN=0x02 RST=0x04
        # PSH=0x08 ACK=0x10 URG=0x20
        bits = {"fin": 0x01, "syn": 0x02, "rst": 0x04,
                "psh": 0x08, "ack": 0x10, "urg": 0x20}

        out = {}
        # (ก) boolean field ตรง ๆ
        have_bool = any(n in tcp for n in names)
        if have_bool:
            for n in names:
                out[f"tcp_{n}"] = 1.0 if tcp.get(n) else 0.0
        else:
            # (ข) parse hex string — รวม flag ทั้งสองทิศทาง (ts=to server,
            # tc=to client) เพราะฝั่ง CICFlowMeter ก็นับรวมทั้ง flow
            merged = 0
            for key in ("tcp_flags", "tcp_flags_ts", "tcp_flags_tc"):
                val = tcp.get(key)
                if isinstance(val, str) and val:
                    try:
                        merged |= int(val, 16)
                    except ValueError:
                        pass
            for n in names:
                out[f"tcp_{n}"] = 1.0 if (merged & bits[n]) else 0.0

        # หมายเหตุ v4.1: ไม่มี derived tcp_syn_no_ack / tcp_rst_no_ack แล้ว
        # (พิสูจน์แล้วว่าเป็น 0 ตลอดตอน live — ดู comment ใต้ TCP_FEATURES)
        return out

    def _handle_suricata_alert(self, raw):
        alert = raw.get("alert", {})
        # sid นี้ไม่ใช่ signature โจมตีจริง เป็น rule ที่เราเขียนเอง
        # (sword-local.rules) ให้ยิง alert ทุก POST request เพื่อ "ดึง"
        # POST body ออกมาเท่านั้น (ดู comment เต็มที่ SWORD_BODY_CAPTURE_SID
        # ต้นไฟล์ + _handle_body_capture_alert) ต้องแยกออกไปก่อนถึง filter
        # อื่นทั้งหมดด้านล่าง ไม่งั้นโดน category="Not Suspicious Traffic"
        # กรองทิ้งไปเงียบๆ (แล้วจะ "จับ POST attack ไม่ได้เลย" แบบที่เจอ)
        if alert.get("signature_id") == SWORD_BODY_CAPTURE_SID:
            self._handle_body_capture_alert(raw)
            return
        cat = alert.get("category", "")
        if cat in {"Generic Protocol Command Decode", "Not Suspicious Traffic"}:
            return
        # กรอง noise ตาม severity — ซ่อน ET INFO (STUN/DNS ปกติ severity 3)
        # เหลือแต่ attack จริง (ดู SURICATA_MIN_SEVERITY) severity 0 = ปิดหมด
        severity = int(alert.get("severity", 3) or 3)
        if SURICATA_MIN_SEVERITY < 1 or severity > SURICATA_MIN_SEVERITY:
            return
        # v5.11: กรอง signature ที่เป็น "informational noise" ทิ้ง แม้ severity
        # จะผ่านเกณฑ์ (ET INFO/POLICY/USER_AGENTS ฯลฯ บางตัว severity=2 แต่ไม่ใช่
        # attack เช่น Steam UA, ET INFO POST cleartext, JA3/TLS fingerprint,
        # DNS lookup ปกติ) — พวกนี้เป็น "traffic ที่ตรง pattern" ไม่ใช่ "การ
        # โจมตี" จึงรกและกลบ alert จริง ดู SURICATA_NOISE_PREFIXES ที่ต้นไฟล์
        sig = alert.get("signature", "") or ""
        if any(sig.startswith(p) for p in SURICATA_NOISE_PREFIXES):
            return
        self.handle_alert({
            "timestamp": time.time(),
            "src_ip": raw.get("src_ip"),
            "dst_ip": raw.get("dest_ip"),
            "dst_port": raw.get("dest_port"),
            "proto": raw.get("proto"),
            "predicted_class": "Suricata Alert",
            "specific_type": alert.get("signature"),   # โชว์ชื่อ rule จริงเวลาที่แสดง
            "is_attack": True,
            "detection_method": "suricata_signature",
        })

    def _handle_body_capture_alert(self, raw):
        """POST body จาก sword-local.rules (sid=SWORD_BODY_CAPTURE_SID) —
        eve.json event_type=="http" ไม่มีช่อง body ให้เลย (ดู comment เต็ม
        ที่ SWORD_BODY_CAPTURE_SID ต้นไฟล์) จึงต้องพึ่ง alert event ที่มี
        http-body-printable/payload-printable แนบมาแทน ตรงนี้ "ไม่" ยิง
        alert ตรงๆ จาก sid นี้ — ต้องเอา body ไปวิ่งผ่าน signature override
        + payload ML เหมือน URI/cookie/UA ก่อน (ผ่าน _check_payload_sources
        ตัวเดียวกับที่ _analyze_http_event ใช้) ถึงจะยิงจริงถ้าเจอ ไม่งั้น
        POST ปกติทุกตัว (login ทั่วไป ฯลฯ) จะโดนแจ้งเตือนหมด
        """
        now = self._event_time(raw)
        src_ip = raw.get("src_ip", "")
        dst_ip = raw.get("dest_ip", "")
        dst_port = raw.get("dest_port", 0)

        # ระหว่าง DoS flood cooldown (ดู _dos_rate_anomaly) ข้ามการวิเคราะห์
        # payload ไปเลย ประหยัด CPU + กัน false positive จาก body ขยะของ
        # เครื่องมือ flood — แค่ "อ่าน" _flood_last เฉยๆ ไม่เรียก
        # _dos_rate_anomaly() ซ้ำ เพราะ event http ของ request เดียวกันนี้
        # ถูกนับอัตราไปแล้วที่ _analyze_http_event เรียกซ้ำจะนับ rate เพี้ยน 2 เท่า
        last_flood = self._flood_last.get((src_ip, dst_ip, dst_port))
        if last_flood is not None and now - last_flood < DOS_FLOOD_COOLDOWN_SECONDS:
            return

        http_data = raw.get("http", {}) or {}
        body = (http_data.get("http_request_body_printable")
                or http_data.get("http_request_body")
                or raw.get("payload_printable")
                or "")
        if not body:
            logger.debug(
                f"[body-capture] alert sid={SWORD_BODY_CAPTURE_SID} ไม่มี "
                "body แนบมา — เช็ค http-body-printable/payload-printable "
                "ใต้ eve-log 'alert' ใน suricata.yaml (ดู setup_suricata.md)"
            )
            return
        status = str(http_data.get("status", "200"))
        # ยิง body ทั้งก้อนเป็น source เดียวไม่พอ — POST body ปกติเป็น
        # application/x-www-form-urlencoded (key1=val1&key2=val2&...) มี
        # field ปกติปนหลายตัว TF-IDF ทั้งก้อนจะเจือจางสัญญาณจน payload ML
        # มองไม่เห็น (พิสูจน์แล้ว: "1' OR '1'='1" เดี่ยวๆ conf=0.999 แต่พอ
        # ต่อกับ "username=admin&...&Login=Login" conf หล่นเหลือ 0.0) —
        # หลักการเดียวกับที่ payload_sources ของ URI/cookie/UA แยก source
        # กันอยู่แล้ว ตรงนี้แยกทีละ field เพิ่มด้วย
        sources = [("post_body", body)]
        for key, val in parse_qsl(body, keep_blank_values=True):
            if val:
                sources.append((f"post_body:{key}", val))
        self._check_payload_sources(raw, sources, status)

    def handle_alert(self, res):
        # ล็อกทั้งฟังก์ชัน — อาจถูกเรียกจากหลาย worker thread พร้อมกันตอน
        # monitor_realtime() ทำงานแบบ producer/consumer ถ้าไม่ล็อก
        # _alert_cache/alert_count/ไฟล์ log จะ race กันได้ (นับเลขซ้ำ,
        # เขียนไฟล์ทับกัน, dedup cache พลาด) I/O ตรงนี้เร็วอยู่แล้วเทียบกับ
        # เวลาที่ ML inference ใช้ ล็อกทั้งก้อนไม่ทำให้ throughput แย่ลง
        with self._alert_lock:
            now = time.time()
            # PortScan ที่มาจากชั้น temporal เป็น "เหตุการณ์ระดับ host คู่
            # หนึ่ง" ไม่ใช่ระดับ port — ถ้าใส่ dst_port ลงใน dedup key จะ
            # ยิง alert ออกมาเป็นร้อยครั้ง (port ละครั้ง) ทั้งที่เป็นการ
            # สแกนครั้งเดียว จึงตัด dst_port ออกจาก key เฉพาะกรณีนี้ และ
            # กดเงียบนานกว่าปกติ (60 วิ = เท่าหน้าต่างที่ใช้นับ)
            if res.get("predicted_class") == "PortScan":
                key = (res.get("src_ip"), res.get("dst_ip"), "PortScan")
                quiet = PORTSCAN_WINDOW_SECONDS
            else:
                key = (res.get("src_ip"), res.get("dst_ip"),
                       res.get("dst_port"), res.get("predicted_class"))
                quiet = 10
            if now - self._alert_cache.get(key, 0) < quiet:
                return
            self._alert_cache[key] = now
            self.alert_count += 1

            predicted = res.get("predicted_class", "Unknown")
            specific = res.get("specific_type") or predicted
            confidence = res.get("confidence", 0.0)
            method = res.get("detection_method", "ml")

            # Header
            print(
                f"🚨 [{self.alert_count}] {res.get('src_ip')} -> "
                f"{res.get('dst_ip')}:{res.get('dst_port')} | "
                f"Attack: {specific} | Method: {method.upper()} | "
                f"Conf: {confidence:.3f}"
            )

            # Log — /var/log ปกติต้อง root ถึงเขียนได้ ถ้ารันแบบ user ธรรมดา
            # (เช่นตอนทดสอบกับ Kali ไม่ได้ sudo) ให้ fallback มาเขียนใน
            # working directory แทน กันสคริปต์ค้าง/พังกลางคันเพราะ log ไม่ได้
            # *** resolve ครั้งเดียวแล้ว cache *** ไม่งั้น mkdir /var/log จะ
            # fail แล้ว warn ซ้ำ "ทุก alert" (รก log มาก) — cache ใน
            # self._log_dir พอครั้งแรก warn ครั้งเดียว
            #
            # เดิม try/except ครอบแค่ mkdir() เท่านั้น ซึ่งพลาดกรณีที่พบจริง:
            # ถ้า ML_LOG_DIR มีอยู่แล้ว (เช่น เคยรันด้วย sudo มาก่อน สร้างไว้
            # เป็นของ root) mkdir(exist_ok=True) จะ "สำเร็จ" เฉยๆ โดยไม่เช็ค
            # สิทธิ์เขียนไฟล์ข้างในเลย แล้วไปพังตอน open() ข้างล่างแทน ซึ่งอยู่
            # นอก try/except — ต้องลอง open จริง (ไม่ใช่แค่ mkdir) ถึงจะรู้ว่า
            # เขียนได้จริงไหม แล้วค่อย cache ผล
            log_dir = getattr(self, "_log_dir", None)
            if log_dir is None:
                log_dir = Path(ML_LOG_DIR)
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with open(log_dir / "ml_log.json", "a"):
                        pass  # ทดสอบว่าเขียนไฟล์ได้จริง ไม่ใช่แค่ mkdir สำเร็จ
                except PermissionError:
                    log_dir = Path("./sword_detection_logs")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    logger.warning(
                        f"ไม่มีสิทธิ์เขียน {ML_LOG_DIR} (รัน sudo หรือ chown ให้ user นี้ถ้าอยากใช้ path เดิม) "
                        f"— fallback มาเขียน log ที่ {log_dir.resolve()} แทน (เตือนครั้งเดียว)"
                    )
                self._log_dir = log_dir
            with open(log_dir / "ml_log.json", "a") as f:
                f.write(json.dumps({
                    "timestamp": now,
                    "src_ip": res.get("src_ip"),
                    "dest_ip": res.get("dst_ip"),
                    "dest_port": res.get("dst_port"),
                    "confidence": f"{confidence:.3f}",
                    "predicted_class": predicted,
                    "specific_attack": specific,
                    "detection_method": method,
                }) + "\n")

    def _process_line(self, line):
        # เฉพาะ json.loads เท่านั้นที่ควรเงียบ — eve.json ที่ Suricata กำลัง
        # เขียนอยู่ (tail -f) อาจโดนอ่านตอนบรรทัดยังเขียนไม่จบ ถือเป็นเรื่อง
        # ปกติของการ tail ไฟล์สด ไม่ใช่บั๊ก
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return

        # ตั้งแต่ตรงนี้ไป "ไม่" ดัก Exception เงียบ ๆ อีกแล้ว (เคยเป็น
        # `except Exception: pass` ครอบทั้งฟังก์ชัน ทำให้บั๊กจริงในนี้ เช่น
        # event รูปแบบผิดปกติที่ทำให้ _analyze_flow/_analyze_http_event
        # โยน exception หายไปเงียบ ๆ ไม่มี log แม้แต่บรรทัดเดียว — ซึ่งทำให้
        # logger.exception("worker พังระหว่างประมวลผล event") ใน
        # monitor_realtime() เป็นโค้ดที่ไม่มีวันทำงานจริงเพราะ exception ไม่
        # เคยลอดไปถึง ปล่อยให้ exception หลุดออกไปให้ worker ของ
        # monitor_realtime() จับแล้ว log แทน ตรงกับที่ตั้งใจไว้ตั้งแต่ต้น)
        event_type = raw.get("event_type")

        if event_type == "alert":
            self._handle_suricata_alert(raw)
            return

        if event_type == "http":
            self._analyze_http_event(raw)
            return

        if event_type == "flow":
            src_ip = raw.get("src_ip", "")
            if src_ip in self.local_ips:
                return
            # กรองเฉพาะปลายทางที่เป็น multicast/broadcast ทิ้งเท่านั้น
            # (MLD, mDNS, SSDP, ...) เพราะนิยามของ "attacker->victim"
            # ไม่มีทางที่ victim จะเป็น multicast group ได้ ไม่ใช่การ
            # เดาจาก protocol — ตั้งใจ "ไม่" กรอง ICMP/ICMPv6 unicast
            # ทิ้ง เพราะ ICMP flood (เช่น Smurf, ping flood) เป็น DoS
            # vector จริงที่มีอยู่จริง ถ้ากรองทิ้งหมดจะทำให้ระบบตาบอด
            # ต่อ attack ประเภทนี้ไปเลย — โมเดลอาจยังตรวจ ICMP flood
            # ได้ไม่แม่น (dataset เทรนไม่มีตัวอย่าง ICMP flood) ซึ่งเป็น
            # ข้อจำกัดที่บันทึกไว้ตรงๆ ในเล่ม ไม่ใช่ปิดบังด้วยการกรองทิ้ง
            if self._is_non_attack_dest(raw.get("dest_ip", "")):
                return
            # v5.11: HOME_NET scoping — วิเคราะห์เฉพาะ flow ที่ปลายทางอยู่
            # ในเครือข่ายที่เราปกป้อง (ดู HOME_NET ที่ต้นไฟล์) traffic ขา
            # ออกไปอินเทอร์เน็ต (เกม/P2P/เว็บ) ของเครื่องในบ้านจะไปเปิด
            # หลายพอร์ตปลายทางที่ IP สาธารณะ ซึ่ง "หน้าตาเหมือน PortScan/
            # DoS" ทั้งที่เป็นการใช้งานปกติ — เป็น false positive ที่เจอจริง
            # ตอนลงเครือข่ายบ้าน (เช่น .51 -> 175.100.59.246:8493) การจำกัด
            # ที่ปลายทางเป็นหลักมาตรฐานของ NIDS ไม่ใช่ hardcode IP
            if not _ip_in_home_net(raw.get("dest_ip", "")):
                return
            self._analyze_flow(raw)

    @staticmethod
    def _is_non_attack_dest(dest_ip: str) -> bool:
        """multicast/broadcast destination = network housekeeping traffic
        (MLD, mDNS, SSDP, ...) — ไม่มีทางเป็น victim ของ attack ได้ตาม
        นิยาม จึงตัดทิ้งก่อนเข้าโมเดลได้อย่างชอบธรรม (ต่างจากการกรองด้วย
        protocol ซึ่งเป็นการเดา ไม่ใช่ตัดจากนิยาม)"""
        if not dest_ip:
            return False
        if dest_ip == "255.255.255.255":
            return True
        try:
            return ipaddress.ip_address(dest_ip.split("%")[0]).is_multicast
        except ValueError:
            return False

    @staticmethod
    def _event_time(raw) -> float:
        """เวลาที่ event 'เกิดจริง' จาก eve.json (raw['timestamp']) เป็น epoch
        วินาที — *ต้องใช้ตัวนี้กับหน้าต่างเวลาเชิงตรวจจับทุกตัว* ไม่ใช่
        time.time() (เวลาที่ประมวลผล) เพราะเวลาโดน flood จะเกิด queue backlog
        แล้ว event ถูกประมวลผลช้ากว่าที่เกิดจริงมาก ถ้าใช้เวลาประมวลผลมาคิด
        'อัตรา' จะเพี้ยน (อัตราถูกยืดออก) ทำให้จับ DoS ช้า/พลาด และ event
        ของ flood หลุดไปถึง payload ML จน false positive — ใช้เวลา event
        จริงทำให้อัตราถูกต้องไม่ว่าจะ backlog แค่ไหน"""
        ts = raw.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts).timestamp()
            except (ValueError, TypeError):
                pass
        return time.time()

    def _bruteforce_repeated(self, src_ip, dst_ip, dst_port, now) -> bool:
        """คืน True ถ้า (src_ip,dst_ip,dst_port) นี้ถูก ML ทายเป็น
        BruteForce candidate ซ้ำอย่างน้อย BRUTEFORCE_MIN_ATTEMPTS ครั้ง
        (รวมครั้งนี้ด้วย) ภายใน BRUTEFORCE_WINDOW_SECONDS วินาทีที่ผ่านมา
        — ดูเหตุผลเต็มๆ ที่ comment เหนือ BRUTEFORCE_MIN_ATTEMPTS ต้นไฟล์
        now = เวลา event จริง (ดู _event_time)"""
        key = (src_ip, dst_ip, dst_port)
        with self._bf_lock:
            dq = self._bf_attempts[key]
            dq.append(now)
            while dq and now - dq[0] > BRUTEFORCE_WINDOW_SECONDS:
                dq.popleft()
            return len(dq) >= BRUTEFORCE_MIN_ATTEMPTS

    def _dos_rate_anomaly(self, src_ip, dst_ip, dst_port, now):
        """ตรวจ DoS แบบ adaptive — ไม่มีเลข "จำนวน" ตายตัว

        แนวคิด: เรียนรู้ baseline ของ "อัตราต่อต้นทาง" ที่บริการ (dst,port)
        เห็นตามปกติแบบออนไลน์ (EWMA ของ mean กับ mean-abs-deviation) แล้ว
        flag ต้นทางที่มีอัตราเป็น outlier ทางสถิติ (robust z > DOS_ROBUST_Z)
        เทียบกับ baseline ของบริการนั้นเอง — ดูเหตุผลเต็มที่ comment เหนือ
        DOS_ROBUST_Z

        ทำไมจับ DoS ช้าได้: ถ้าบริการปกติเงียบ (baseline ~0.2 req/s) แล้ว
        ผู้โจมตีส่งช้า ๆ แค่ 4 req/s ก็ยังเป็น z สูงมากเทียบ baseline ->
        จับได้ โดยไม่ต้องรอให้ยิงครบจำนวนใด ๆ และปรับตามแต่ละบริการเอง

        now = เวลา event จริง (ดู _event_time)
        คืน (is_flood: bool, rate: float)
        """
        fkey = (src_ip, dst_ip, dst_port)
        skey = (dst_ip, dst_port)
        with self._dos_lock:
            dq = self._dos_win[fkey]
            dq.append(now)
            while dq and now - dq[0] > DOS_RATE_WINDOW_SECONDS:
                dq.popleft()

            # ── cooldown: เพิ่งจับ flood ของ (src,dst,port) นี้ไปเมื่อกี้ ──
            # ถือว่ายังโจมตีอยู่ ลัดวงจรทันที (ข้าม ML/payload) จนกว่าจะเงียบ
            # เกิน DOS_FLOOD_COOLDOWN_SECONDS — กัน XSS/SQLi false positive
            # จาก URL มั่ว ๆ ของเครื่องมือ flood + ลด backlog
            last = self._flood_last.get(fkey)
            if last is not None and now - last < DOS_FLOOD_COOLDOWN_SECONDS:
                self._flood_last[fkey] = now
                return True, len(dq) / DOS_RATE_WINDOW_SECONDS
            rate = len(dq) / DOS_RATE_WINDOW_SECONDS  # req/s เฉลี่ยของต้นทางนี้ต่อบริการนี้

            st = self._svc_stats[skey]
            mean, mad = st["mean"], st["mad"]

            # robust z-score เทียบ baseline (เริ่มจาก prior ต่ำ ดู DOS_PRIOR_RATE)
            z = (rate - mean) / (mad + 1e-9)
            is_flood = (z > DOS_ROBUST_Z) and (rate >= DOS_MIN_RATE)

            if is_flood:
                self._flood_last[fkey] = now   # เริ่มนับ cooldown

            # ── อัปเดต baseline: ตามเวลา (ไม่ใช่ทุก event) + poisoning guard ──
            # อัปเดตอย่างมาก 1 ครั้งต่อ DOS_BASELINE_UPDATE_INTERVAL วินาที
            # ต่อบริการ (ดู comment เหนือค่านั้น) เพื่อไม่ให้ event volume ของ
            # flood ดึง baseline ตาม + รับเฉพาะค่าที่อยู่ในเกณฑ์ปกติ (guard,
            # <= mean+3·mad) traffic โจมตีที่พุ่งสูงจึงไม่เข้ามาอัปเดต baseline
            # -> baseline คงต่ำ -> z พุ่งถึงเกณฑ์ทันที (แก้ cold-start poisoning
            # ที่ทำให้ GoldenEye หลุดมาก่อนหน้านี้)
            st["n"] = st.get("n", 0) + 1
            if now - st.get("last_update", 0.0) >= DOS_BASELINE_UPDATE_INTERVAL:
                st["last_update"] = now
                if rate <= mean + 3.0 * (mad + 1e-9):
                    st["mean"] = (1 - DOS_EWMA_ALPHA) * mean + DOS_EWMA_ALPHA * rate
                    st["mad"] = ((1 - DOS_EWMA_ALPHA) * mad
                                 + DOS_EWMA_ALPHA * abs(rate - st["mean"]))
            return is_flood, rate

    def _load_dos_baseline(self):
        """โหลด baseline ต่อบริการที่ profile ไว้จากรอบก่อน (ถ้ามี) — ยิ่งรัน
        นาน/หลายรอบ baseline ยิ่งแม่น เหมือน anomaly-IDS มาตรฐานที่เรียนรู้
        พฤติกรรมปกติของแต่ละบริการไว้ก่อน (persist ข้ามรอบ)"""
        path = os.path.join(self.model_dir, DOS_BASELINE_FILE)
        if not os.path.exists(path):
            return
        try:
            data = json.load(open(path))
            for k, st in data.items():
                dst_ip, port = k.rsplit("|", 1)
                self._svc_stats[(dst_ip, int(port))] = {
                    "mean": float(st["mean"]), "mad": float(st["mad"]),
                    "n": int(st.get("n", 0))}
            logger.info(f"DoS baseline: โหลด profile ของ {len(data)} บริการ จากรอบก่อน")
        except Exception as e:
            logger.warning(f"DoS baseline: โหลดไม่สำเร็จ ({e}) — เริ่มด้วย prior")

    def _save_dos_baseline(self):
        """บันทึก baseline ต่อบริการลงไฟล์ (เรียกตอนปิดโปรแกรม) เพื่อใช้รอบหน้า"""
        path = os.path.join(self.model_dir, DOS_BASELINE_FILE)
        try:
            with self._dos_lock:
                data = {f"{dst}|{port}": {"mean": st["mean"], "mad": st["mad"],
                                           "n": st["n"]}
                        for (dst, port), st in self._svc_stats.items()}
            json.dump(data, open(path, "w"), indent=2)
            logger.info(f"DoS baseline: บันทึก profile ของ {len(data)} บริการ -> {path}")
        except Exception as e:
            logger.warning(f"DoS baseline: บันทึกไม่สำเร็จ ({e})")

    def _emit_dos_flood(self, raw, rate):
        self.handle_alert({
            "src_ip": raw.get("src_ip", ""),
            "dst_ip": raw.get("dest_ip", ""),
            "dst_port": raw.get("dest_port", 0),
            "predicted_class": "DoS",
            "specific_type": (f"DoS rate anomaly ({rate:.1f} req/s, "
                               f"outlier vs learned baseline)"),
            "confidence": 0.95,
            "is_attack": True,
            "detection_method": "adaptive_rate_anomaly",
            "all_scores": {},
        })

    def _portscan_fanout(self, raw, now) -> int:
        """บันทึก flow นี้ลงหน้าต่างเวลา แล้วคืน "จำนวน port ไม่ซ้ำที่ src
        นี้แตะไปยัง dst นี้แบบ scan-like ภายใน PORTSCAN_WINDOW_SECONDS"

        นับเฉพาะ flow ที่ปลายทางแทบไม่ตอบอะไรกลับมาเลย (RST หรือเงียบ)
        = ไม่เคยมี session จริง ซึ่งเป็นลักษณะของการ probe ส่วน traffic
        ปกติที่แตะหลาย port จะมี payload กลับมาเสมอ จึงไม่ถูกนับ
        (ดูเหตุผลเต็มที่ comment เหนือ PORTSCAN_MIN_DISTINCT_PORTS)

        คืน 0 ถ้า flow นี้ไม่ scan-like (ไม่ต้องนับ ไม่ต้องเช็คต่อ)
        """
        if raw.get("proto") not in ("TCP", "UDP"):
            return 0
        flow = raw.get("flow", {}) or {}
        bwd_bytes = int(flow.get("bytes_toclient", 0) or 0)
        bwd_pkts = int(flow.get("pkts_toclient", 0) or 0)
        if not (bwd_bytes <= PORTSCAN_SCANLIKE_MAX_BWD_BYTES
                and bwd_pkts <= PORTSCAN_SCANLIKE_MAX_BWD_PKTS):
            return 0

        key = (raw.get("src_ip", ""), raw.get("dest_ip", ""))
        port = raw.get("dest_port", 0)
        with self._ps_lock:
            dq = self._ps_events[key]
            dq.append((now, port))
            while dq and now - dq[0][0] > PORTSCAN_WINDOW_SECONDS:
                dq.popleft()
            return len({p for _, p in dq})

    def _analyze_flow(self, raw):
        # เวลา event จริง (ไม่ใช่เวลาประมวลผล) — สำคัญมากตอน backlog ดู _event_time
        now = self._event_time(raw)
        # ── ชั้น adaptive: DoS rate anomaly (ทำก่อนสุด) ─────────────
        # อัตราของต้นทางเป็น outlier เทียบ baseline ที่บริการนี้เรียนรู้ =
        # DoS (ปรับตามเครือข่ายเอง จับได้ทั้ง flood เร็ว/ช้า ไม่ต้องรอครบ
        # จำนวน) ยิงแล้วข้าม ML ประหยัด CPU ตอนโดน flood — ดู _dos_rate_anomaly
        if ENABLE_DOS_RATE:
            _dos, _rate = self._dos_rate_anomaly(
                raw.get("src_ip", ""), raw.get("dest_ip", ""),
                raw.get("dest_port", 0), now)
            if _dos:
                self._emit_dos_flood(raw, _rate)
                return

        # ── ชั้น temporal: PortScan (ทำก่อน ML โดยตั้งใจ) ────────────
        # ทำงาน "ทุก flow" ไม่ว่า ML จะว่าอย่างไร เพราะสัญญาณของ port scan
        # ไม่ได้อยู่ใน flow เดี่ยว ๆ เลย (ดู comment เหนือ
        # PORTSCAN_MIN_DISTINCT_PORTS) — ถ้ารอให้ ML ฟันธงก่อนค่อยนับ ก็จะ
        # ไม่มีวันนับได้ เพราะ ML มองเห็นทีละ flow เท่านั้น
        #
        # เหตุผลที่วางไว้ "ก่อน" predict() ไม่ใช่หลัง: การนับ fan-out เป็น
        # การบวกเลขในหน่วยความจำ (ไมโครวินาที) ส่วน predict() ต้องเรียก
        # CalibratedClassifierCV 3 task x 3 model x 5 fold = 45 sub-model
        # ต่อ 1 flow (มิลลิวินาที) — ตอนโดนสแกน 1000 port จะมี flow ไหลเข้า
        # มา 1000 อันรวดเดียว ซึ่งเป็นจังหวะเดียวกับที่ queue backlog พุ่ง
        # พอดี ถ้ารู้แล้วว่า alert ไปแล้วในหน้าต่างนี้ ก็ข้าม ML ไปเลย
        # ประหยัด CPU ได้มหาศาลตรงจุดที่ต้องการที่สุด
        n_ports = self._portscan_fanout(raw, now) if ENABLE_PORTSCAN_FANOUT else 0
        if n_ports >= PORTSCAN_MIN_DISTINCT_PORTS:
            src_ip = raw.get("src_ip", "")
            dst_ip = raw.get("dest_ip", "")
            ps_key = (src_ip, dst_ip, "PortScan")
            with self._alert_lock:
                recent = (time.time() - self._alert_cache.get(ps_key, 0)
                          < PORTSCAN_WINDOW_SECONDS)
            if recent:
                # แจ้งเตือนไปแล้วสำหรับคู่ host นี้ในหน้าต่างนี้ — ไม่ต้อง
                # เสีย CPU รัน ML ซ้ำอีก 999 รอบระหว่างสแกนชุดเดียวกัน
                return
            res = self.predict(pd.DataFrame([self._build_feature_row(raw)]))
            ml_agrees = res.get("predicted_class") == "PortScan"
            # ML เห็นด้วย = มั่นใจสูงกว่า (สองชั้นตรงกัน) แต่ถึง ML จะไม่
            # เห็นด้วยก็ยัง alert เพราะ fan-out ระดับนี้คือนิยามของ scan
            conf = 0.99 if ml_agrees else 0.90
            self.handle_alert({
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "dst_port": raw.get("dest_port", 0),
                "predicted_class": "PortScan",
                "specific_type": (f"Port Scan ({n_ports} distinct ports in "
                                   f"{int(PORTSCAN_WINDOW_SECONDS)}s)"),
                "confidence": conf,
                "is_attack": True,
                "detection_method": ("ml+temporal_fanout" if ml_agrees
                                      else "temporal_fanout"),
                "all_scores": res.get("all_scores", {}),
            })
            return

        row = self._build_feature_row(raw)
        res = self.predict(pd.DataFrame([row]))
        if res["is_attack"]:
            pred = res["predicted_class"]
            # PortScan ที่ ML ทายทีละ flow แต่ fan-out ไม่ถึงเกณฑ์ (มาถึง
            # ตรงนี้ได้แปลว่า distinct-port < PORTSCAN_MIN_DISTINCT_PORTS)
            # = ไม่ใช่ scan จริง เป็นแค่ flow สั้น ๆ ที่หน้าตาคล้าย scan
            # (เช่น connection ของ sqlmap/GoldenEye ที่ยิง port เดียว) — กัน
            # false positive เพราะ port scan ตามนิยามต้องแตะหลาย port
            if pred == "PortScan":
                return
            # DoS: per-flow ML ไม่น่าเชื่อถือเช่นเดียวกับ PortScan — DoS เป็น
            # ปรากฏการณ์ "เชิงปริมาณ/อัตรา" ไม่ใช่ flow เดี่ยว flow เดียวที่
            # ML ทายว่า DoS มักเป็น false positive (เช่น probe ของ nmap ไป
            # port ที่เปิด หน้าตา flow คล้าย DoS flow ใน CICIDS) — DoS จริง
            # ตรวจด้วยชั้น adaptive rate anomaly (ทำไปแล้วต้นฟังก์ชัน) ซึ่ง
            # เชื่อถือได้และปรับตัวได้ จึง "ไม่" ยิง DoS จาก per-flow ML ตรงนี้
            # (โมเดล dos ยังเทรน/เก็บไว้เทียบผลในเล่ม แค่ไม่ใช้ยิง alert สด
            # หลักการเดียวกับที่ปิด webattack flow-ML) — เงื่อนไขนี้ผูกกับ
            # ENABLE_DOS_RATE โดยตั้งใจ: ถ้าปิด adaptive rate (ไปเทียบผล ML
            # ล้วนๆ ตามที่ขอ) ก็ต้อง "ปล่อย" ให้ per-flow ML ทาย DoS ได้เอง
            # แทน ไม่งั้นปิด adaptive rate แล้วจะกลายเป็น "ไม่ตรวจ DoS เลย"
            # ทั้งที่ตั้งใจจะสลับไปใช้ ML ล้วนต่างหาก (ดู comment เหนือ
            # ENABLE_DOS_RATE ต้นไฟล์ — ผลคือ false positive เยอะขึ้นจริง
            # เช่น nmap probe ไป port เปิดอาจโดนแปะ DoS ตามที่อธิบายไว้)
            if pred == "DoS" and ENABLE_DOS_RATE:
                return
            if pred == "BruteForce" and ENABLE_BRUTEFORCE_GATE and not \
                    self._bruteforce_repeated(raw.get("src_ip", ""),
                                               raw.get("dest_ip", ""),
                                               raw.get("dest_port", 0), now):
                return
            specific = _PREDICTED_CLASS_DISPLAY.get(pred, pred)
            res.update({
                "specific_type": specific,
                "src_ip": raw.get("src_ip", ""),
                "dst_ip": raw.get("dest_ip", ""),
                "dst_port": raw.get("dest_port", 0),
                "detection_method": "ml",
            })
            self.handle_alert(res)

    def _run_payload_ml(self, uri: str) -> tuple:
        """
        Payload ML Pipeline 2: TF-IDF → XGBoost multiclass
        คืนค่า (specific_type, confidence) หรือ (None, 0.0)
        """
        if not self.payload_vec or not self.payload_xgb:
            return None, 0.0
        decoded = unquote(uri).lower()
        vec = self.payload_vec.transform([decoded])
        probs = self.payload_xgb.predict_proba(vec)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
        MAP = {1: "SQL Injection", 2: "Cross-Site Scripting (XSS)"}
        if pred != 0 and conf >= self.payload_threshold:
            return MAP[pred], conf
        return None, 0.0

    def _get_all_payloads(self, raw: dict) -> list:
        """
        รวบรวม payloads ทั้งหมดจาก HTTP event ที่อาจมีการโจมตี
        - URI (GET/POST parameters)
        - request_body (POST body)
        - User-Agent (บางครั้งมี XSS)
        - Referer
        - Cookie

        Suricata eve.json จะมี request_body เฉพาะเมื่อตั้ง
        app-layer.protocols.http.libhtp.default-config.request-body-limit > 0
        """
        http_data = raw.get("http", {}) or {}
        sources = []

        uri = http_data.get("url", "")
        if uri:
            sources.append(("uri", uri))
            # แยก query string เป็นทีละ parameter เพิ่ม — URL จริงมี path
            # prefix (เช่น "/vulnerabilities/sqli/") ปนกับ query string ทำให้
            # TF-IDF ทั้งก้อนเจือจางสัญญาณจนคะแนนหล่นเหลือ 0 (พิสูจน์แล้ว:
            # "id=1' OR '1'='1" เดี่ยวๆ conf=0.999 แต่พอมี path prefix ปนมา
            # ด้วยเป็น "/vulnerabilities/sqli/?id=1' OR '1'='1&Submit=Submit"
            # conf หล่นเหลือ 0.0 ทั้งที่ query string เดียวกันเป๊ะ) —
            # หลักการเดียวกับที่ POST body ใช้แยกทีละ field (ดู
            # _handle_body_capture_alert) query string เป็น
            # x-www-form-urlencoded รูปแบบเดียวกับ POST body พอดี
            if "?" in uri:
                query = uri.split("?", 1)[1]
                for key, val in parse_qsl(query, keep_blank_values=True):
                    if val:
                        sources.append((f"uri_param:{key}", val))

        # POST body — Suricata ใช้ key "request_body" หรือ "http.request_body"
        body = http_data.get("request_body", "")
        if not body and "http" in raw:
            body = raw.get("http_request_body", "")
        if body:
            sources.append(("body", body))

        # User-Agent — บาง payload XSS แอบใน header แต่ UA ปกติของเบราว์เซอร์/
        # scanner ทั่วไป (Chrome, nmap NSE ฯลฯ) ก็มี "(" ")" ";" เป็นไวยากรณ์
        # มาตรฐานอยู่แล้ว (เช่น "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...")
        # เช็คด้วยแค่อักขระพวกนี้จะ false positive กับ UA ปกติแทบทุกตัว
        # เปลี่ยนมาเช็คด้วย keyword ชุดเดียวกับ signature-based detection
        # (_XSS_PATTERNS/_SQLI_PATTERNS) แทน ให้ตรงเฉพาะ pattern โจมตีจริงๆ
        ua = http_data.get("http_user_agent", "")
        if ua:
            ua_low = ua.lower()
            if any(kw in ua_low for kw, _ in _XSS_PATTERNS) or \
               any(kw in ua_low for kw, _ in _SQLI_PATTERNS):
                sources.append(("user_agent", ua))

        # Cookie
        cookie = http_data.get("cookie", "")
        if cookie:
            sources.append(("cookie", cookie))

        return sources

    def _check_payload_sources(self, raw, payload_sources, status) -> bool:
        """ตรวจ payload_sources (list ของ (src_name, payload)) ด้วย
        signature override ก่อน แล้วค่อย payload ML — แยกออกมาจาก
        _analyze_http_event เดิม เพื่อให้ _handle_body_capture_alert()
        (POST body ที่ได้จาก Suricata alert event แทน http event — ดู
        comment เต็มที่ตัวนั้น) เรียกใช้ตรรกะเดียวกันได้โดยไม่ต้องก็อปโค้ด

        คืน True ถ้ายิง alert ไปแล้ว (ผู้เรียกควร return ทันที) ไม่งั้น False
        """
        # ── ด่าน 1: Signature override (XSS/SQLi) ──────────────
        # ปิดได้ด้วย USE_SIGNATURE_OVERRIDE=False เพื่อให้ payload ML เป็น
        # คนตัดสินเว็บล้วน ๆ (ดู comment ที่ตัวแปรต้นไฟล์) — Suricata
        # signature ไม่เกี่ยวกับด่านนี้ ยังทำงานปกติผ่าน _handle_suricata_alert
        if USE_SIGNATURE_OVERRIDE:
            for src_name, payload in payload_sources:
                specific_type, sig_conf = detect_web_signature(payload, status)
                if specific_type:
                    self.handle_alert({
                        "predicted_class": "WebAttack",
                        "specific_type": specific_type,
                        "confidence": sig_conf,
                        "is_attack": True,
                        "model_conf": {},
                        "all_scores": {"WebAttack": sig_conf},
                        "all_per_model": {},
                        "detection_method": "signature_override",
                        "notes": f"matched in {src_name}",
                        "src_ip": raw.get("src_ip", ""),
                        "dst_ip": raw.get("dest_ip", ""),
                        "dst_port": raw.get("dest_port", 0),
                    })
                    return True

        # ── ด่าน 2: Payload ML (TF-IDF + XGBoost) ─────────────
        # ยิง payload "แต่ละแหล่งแยกกัน" แล้วเลือกอันที่มั่นใจสูงสุด แทนการ
        # เอาทุกแหล่งมาต่อกันเป็นสตริงเดียว (ซึ่งเจือจางสัญญาณ ทำให้ conf ต่ำ
        # เช่น SQLi ใน uri สั้น ๆ พอต่อกับ cookie ยาว ๆ ความมั่นใจก็ลด)
        # วิธีนี้ได้ทั้ง conf ที่สูงขึ้น และรู้ว่าเจอในแหล่งไหน
        best_spec, best_conf, best_src = None, 0.0, None
        for src_name, payload in payload_sources:
            spec, conf = self._run_payload_ml(payload)
            if spec and conf > best_conf:
                best_spec, best_conf, best_src = spec, conf, src_name
        if best_spec:
            self.handle_alert({
                "predicted_class": "WebAttack",
                "specific_type": best_spec,
                "confidence": best_conf,
                "is_attack": True,
                "model_conf": {},
                "all_scores": {"WebAttack": best_conf},
                "all_per_model": {},
                "detection_method": "pipeline_2_payload_ml",
                "notes": f"detected in {best_src}",
                "src_ip": raw.get("src_ip", ""),
                "dst_ip": raw.get("dest_ip", ""),
                "dst_port": raw.get("dest_port", 0),
            })
            return True
        return False

    def _analyze_http_event(self, raw):
        http_data = raw.get("http", {})
        if not http_data:
            return

        # v5.11: HOME_NET scoping — สนใจเฉพาะ HTTP ที่ยิงเข้ามาหา asset ที่เรา
        # ปกป้อง ถ้าเป็นเครื่องในบ้านออกไปเปิดเว็บข้างนอก (ปลายทางเป็น IP
        # สาธารณะ) เราไม่ใช่คนปกป้องเซิร์ฟเวอร์นั้น จึงไม่ควรเอา request ของ
        # ผู้ใช้ไปตัดสินเป็น attack/DoS (กัน FP + ไม่ให้ browsing ปกติไป
        # poison baseline ของ DoS) — เหตุผลเดียวกับฝั่ง flow
        if not _ip_in_home_net(raw.get("dest_ip", "")):
            return

        now = self._event_time(raw)   # เวลา event จริง (ดู _event_time)
        # ── ชั้น adaptive: DoS rate anomaly (HTTP flood เช่น GoldenEye) ──
        # request รัว ๆ ไปบริการเดียวจน "อัตราเป็น outlier" เทียบ baseline
        # = DoS — per-flow ML มองไม่เห็นเพราะแต่ละ request เหมือน GET ปกติ
        # ใช้ state เดียวกับ flow event (ดู _dos_rate_anomaly) — cooldown ใน
        # นั้นจะลัดวงจร request ของ flood ก่อนถึง payload ML กัน XSS FP จาก
        # URL มั่ว ๆ ของ GoldenEye
        if ENABLE_DOS_RATE:
            _dos, _rate = self._dos_rate_anomaly(
                raw.get("src_ip", ""), raw.get("dest_ip", ""),
                raw.get("dest_port", 0), now)
            if _dos:
                self._emit_dos_flood(raw, _rate)
                return

        status = str(http_data.get("status", "200"))
        payload_sources = self._get_all_payloads(raw)

        if self._check_payload_sources(raw, payload_sources, status):
            return

        # ── ด่าน 3: Flow ML (Brute Force) ──────────────────────
        row = self._build_feature_row(raw, http_data=http_data)
        res = self.predict(pd.DataFrame([row]))
        if res["is_attack"]:
            pred = res["predicted_class"]
            # event นี้เป็น HTTP request ที่ "สำเร็จ" (มี http_data) — โดย
            # นิยามแล้วไม่ใช่ port scan (scan ไม่เคยทำ HTTP request จบ) ดังนั้น
            # ถ้า flow ML ทายว่า PortScan ตรงนี้ = false positive แน่นอน
            # (เช่น connection ของ sqlmap/GoldenEye ไป port 8080) — ตัดทิ้ง
            if pred == "PortScan":
                return
            # DoS จาก per-flow ML ตรงนี้ก็ไม่น่าเชื่อถือเช่นกัน — DoS ตรวจด้วย
            # adaptive rate anomaly (ต้นฟังก์ชัน) ไม่ใช่ flow เดี่ยว (ดูเหตุผล
            # เต็มที่ _analyze_flow) — ผูกกับ ENABLE_DOS_RATE เหมือนกัน (ดู
            # comment เต็มที่ _analyze_flow)
            if pred == "DoS" and ENABLE_DOS_RATE:
                return
            if pred == "BruteForce" and ENABLE_BRUTEFORCE_GATE and not \
                    self._bruteforce_repeated(raw.get("src_ip", ""),
                                               raw.get("dest_ip", ""),
                                               raw.get("dest_port", 0), now):
                return
            specific = _PREDICTED_CLASS_DISPLAY.get(pred, pred)
            res.update({
                "specific_type": specific,
                "src_ip": raw.get("src_ip", ""),
                "dst_ip": raw.get("dest_ip", ""),
                "dst_port": raw.get("dest_port", 0),
                "detection_method": "pipeline_1_flow_ml",
            })
            self.handle_alert(res)

    def monitor_realtime(self, path: str, num_workers: int = None,
                          queue_size: int = None):
        """
        อ่าน Suricata eve.json แบบ real-time (เหมือน `tail -f`) แบบ
        producer/consumer multi-thread:
        - thread หลัก (producer) มีหน้าที่แค่ tail ไฟล์ + จัดการ log
          rotation + ใส่บรรทัดที่อ่านได้ลง queue เท่านั้น ทำงานเบา ไม่ค้าง
          รอ ML inference เลย
        - worker thread (consumer) หลายตัวดึงบรรทัดจาก queue มาเข้า
          _process_line() (signature -> payload-ML -> flow ML) พร้อมกัน
          หลาย event ได้จริง เพราะ numpy/XGBoost ปล่อย GIL ระหว่างคำนวณ

        เหตุผลที่ต้องทำแบบนี้: eve.json ดักทั้งวง LAN ไม่ใช่แค่ traffic ที่
        ทดสอบ ถ้าประมวลผลทีละ event แบบ sequential (เดิม) แล้ว event ไหลเข้า
        เร็วกว่าที่ ML ตามทัน จะเกิด backlog สะสมโดยไม่มีใครรู้ (status log
        ก็ไม่ขึ้นด้วยเพราะ loop ไม่เคยว่างเลย) — แบบใหม่นี้ producer เบาพอ
        ที่จะ log สถานะ + เห็นขนาด queue (backlog) ได้ตลอดเวลา ไม่ว่า
        worker จะตามทันหรือไม่

        queue เป็น bounded queue (มี maxsize) เพื่อกัน memory บวมไม่รู้จบ
        ถ้า backlog ใหญ่เกินไป — producer จะ "รอ" (blocking put) ให้ worker
        ตามทันก่อน เป็น backpressure ตามธรรมชาติ ไม่ drop event ทิ้ง

        ต้องตั้งค่า Suricata ให้ output eve.json ครบ (ดู setup_suricata.md
        ที่แนบมาด้วย) ก่อนไฟล์นี้จะมี event ให้อ่าน

        Usage: python hybrid_nids.py --realtime /var/log/suricata/eve.json --workers 8
        """
        num_workers = num_workers or min(8, (os.cpu_count() or 4))
        queue_size = queue_size or num_workers * 200

        stop_event = threading.Event()
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            stop_event.set()
            print("\n[monitor_realtime] ได้รับ Ctrl+C กำลังหยุด (รอ worker เก็บงานที่ค้างอยู่)...")

        signal.signal(signal.SIGINT, _handle_sigint)

        p = Path(path)
        logger.info(f"[monitor_realtime] SWORD v{VERSION} เริ่มตรวจสอบ {p}")
        logger.info(f"[monitor_realtime] Worker threads: {num_workers} | Queue size: {queue_size}")
        logger.info(f"[monitor_realtime] Flow ML tasks ที่ใช้งานจริง: "
                    f"{list(self.models_ovr.keys())} "
                    f"(webattack ปิดโดยตั้งใจ ใช้ payload-ML+signature แทน)")
        logger.info(f"[monitor_realtime] Payload ML (XSS/SQLi): "
                    f"{'พร้อม' if self.payload_xgb is not None else 'ไม่พบโมเดล — ใช้ signature-only'}")

        while not p.exists() and not stop_event.is_set():
            logger.warning(f"[monitor_realtime] ยังไม่พบไฟล์ {p} "
                            f"— รอ Suricata สร้างไฟล์อยู่ (Ctrl+C เพื่อยกเลิก)")
            time.sleep(2)
        if stop_event.is_set():
            signal.signal(signal.SIGINT, prev_handler)
            return

        line_queue = queue.Queue(maxsize=queue_size)
        processed_lock = threading.Lock()
        counters = {"processed": 0}

        def _worker():
            while True:
                item = line_queue.get()
                if item is None:  # sentinel — สัญญาณให้ worker หยุด
                    line_queue.task_done()
                    return
                try:
                    self._process_line(item)
                except Exception:
                    logger.exception("[monitor_realtime] worker พังระหว่างประมวลผล event")
                with processed_lock:
                    counters["processed"] += 1
                line_queue.task_done()

        workers = [threading.Thread(target=_worker, name=f"sword-worker-{i}", daemon=True)
                   for i in range(num_workers)]
        for w in workers:
            w.start()

        def _log_status():
            with processed_lock:
                n = counters["processed"]
            qsize = line_queue.qsize()
            # แยกให้ชัดระหว่าง "ระบบตรวจจับไม่เจออะไร" (ทำงานปกติ) กับ
            # "ไม่มีข้อมูลเข้ามาให้ตรวจเลย" (Suricata ไม่ทำงาน) — สองอย่าง
            # นี้หน้าตาเหมือนกันใน log เดิม แต่แก้คนละทางกันโดยสิ้นเชิง
            if n == 0:
                try:
                    _age = time.time() - os.stat(p).st_mtime
                    _hint = f"eve.json ถูกเขียนล่าสุดเมื่อ {_age:.0f} วินาทีที่แล้ว"
                except OSError:
                    _hint = "อ่านสถานะ eve.json ไม่ได้"
                logger.warning(
                    f"[monitor_realtime] ⚠️  ยังไม่ได้รับ event ใดเลย ({_hint})\n"
                    "    นี่ไม่ใช่ปัญหาของโมเดล — คือไม่มีข้อมูลเข้ามาให้ตรวจ\n"
                    "    เช็คตามลำดับ:\n"
                    "      1) Suricata รันอยู่ไหม : ps aux | grep '[s]uricata'\n"
                    "      2) ไฟล์โตขึ้นไหม        : ls -la " + str(p) + " ; sleep 10 ; ls -la " + str(p) + "\n"
                    "      3) เขียนไฟล์นี้จริงไหม   : grep -A3 'eve-log' /etc/suricata/suricata.yaml | head\n"
                    "      4) BPF filter กรองทิ้งหมดหรือเปล่า (ดู argument ตอนสตาร์ท Suricata)"
                )
                return
            backlog_note = f" ⚠️ queue ค้าง {qsize}/{queue_size} (ตามไม่ทัน)" if qsize > queue_size * 0.5 else ""
            logger.info(f"[monitor_realtime] ทำงานปกติ | ประมวลผลแล้ว {n:,} events | "
                        f"queue ค้างอยู่ {qsize} | alert รวม {self.alert_count}{backlog_note}")

        # ── ตรวจสถานะไฟล์ eve.json ก่อนเริ่มอ่าน ───────────────────
        # ถ้าไฟล์ไม่ถูกเขียนมานานแล้ว แปลว่า Suricata ไม่ได้รันอยู่ ซึ่งเป็น
        # สาเหตุที่พบบ่อยที่สุดของอาการ "ประมวลผล 0 events" — บอกตั้งแต่
        # วินาทีแรก ดีกว่าปล่อยให้รอหลายนาทีแล้วมาเข้าใจผิดว่า ML พัง
        try:
            _st = os.stat(p)
            _age = time.time() - _st.st_mtime
            logger.info(f"[monitor_realtime] eve.json ขนาด {_st.st_size:,} bytes "
                        f"| ถูกเขียนล่าสุดเมื่อ {_age:.0f} วินาทีที่แล้ว")
            if _age > 120:
                logger.warning(
                    "[monitor_realtime] ⚠️  eve.json ไม่ถูกเขียนมานานกว่า 2 นาที "
                    "— น่าจะไม่มี Suricata รันอยู่\n"
                    "    ตรวจ : ps aux | grep '[s]uricata'\n"
                    "    เริ่ม : sudo suricata -c /etc/suricata/suricata.yaml -i enp0s8 -D \\\n"
                    "                \"host 192.168.1.52 or host 192.168.1.54\"\n"
                    "    (ถ้าเคย systemctl disable suricata ไว้ พอ reboot จะไม่มีอะไรสตาร์ทให้)"
                )
        except OSError:
            pass

        last_status = time.time()
        f = open(p, "r", encoding="utf-8", errors="replace")
        f.seek(0, os.SEEK_END)  # เริ่มอ่านจาก event ใหม่เท่านั้น ไม่ backfill ของเก่าในไฟล์

        try:
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    # เช็ค log rotation (suricata/logrotate อาจสร้างไฟล์ใหม่แทนที่)
                    try:
                        if os.stat(p).st_ino != os.fstat(f.fileno()).st_ino:
                            logger.info("[monitor_realtime] ตรวจพบ log rotation กำลังเปิดไฟล์ใหม่...")
                            f.close()
                            f = open(p, "r", encoding="utf-8", errors="replace")
                    except FileNotFoundError:
                        pass
                else:
                    line = line.strip()
                    if line:
                        # producer เบามาก (แค่ I/O) ไม่ต้องรอ ML — ถ้า queue
                        # เต็มจริงๆ (worker ตามไม่ทันหนักมาก) จะ block รอ
                        # ตรงนี้แหละ เป็น backpressure กันความจำบวม
                        line_queue.put(line)

                # log สถานะทุก 60 วิ "ไม่ว่า producer จะ busy หรือ idle" —
                # ต่างจากเวอร์ชันเดิมที่ log เฉพาะตอน idle เท่านั้น ทำให้
                # ตอน backlog หนักๆ (producer ไม่เคยว่าง) ไม่เห็น status เลย
                if time.time() - last_status > 60:
                    _log_status()
                    last_status = time.time()
        finally:
            f.close()
            for _ in workers:
                line_queue.put(None)
            for w in workers:
                w.join(timeout=30)
            signal.signal(signal.SIGINT, prev_handler)
            with processed_lock:
                n = counters["processed"]
            logger.info(f"[monitor_realtime] หยุดแล้ว | ประมวลผลรวม {n:,} events | "
                        f"alert รวม {self.alert_count}")
            self._save_dos_baseline()   # บันทึก baseline ที่เรียนรู้ไว้ใช้รอบหน้า
            # รายงานว่าใช้ flow.start/end คำนวณ duration ได้กี่ % — ถ้า
            # fallback ไป flow.age (หยาบระดับวินาที) เยอะ แปลว่า Suricata
            # ไม่ได้ส่ง start/end มา ซึ่งจะทำให้ feature ที่อิง duration
            # เพี้ยนอีก ต้องรู้ ไม่ใช่ปล่อยพังเงียบ ๆ
            tot, fb = HybridNIDS._dur_total, HybridNIDS._dur_fallback
            if tot:
                pct = 100.0 * fb / tot
                if pct > 5:
                    logger.warning(
                        f"[monitor_realtime] ⚠️  duration: ต้อง fallback ไปใช้ "
                        f"flow.age (หยาบระดับวินาที) {fb:,}/{tot:,} ครั้ง ({pct:.1f}%)\n"
                        "    แปลว่า Suricata ไม่ได้ส่ง flow.start/flow.end มาด้วย\n"
                        "    feature ที่อิง duration จะเพี้ยนเทียบกับตอนเทรน\n"
                        "    ตรวจ: grep -A8 'eve-log' /etc/suricata/suricata.yaml")
                else:
                    logger.info(
                        f"[monitor_realtime] duration: ใช้ flow.start/end "
                        f"(ละเอียดไมโครวินาที) {tot - fb:,}/{tot:,} ครั้ง "
                        f"({100 - pct:.1f}%) — ตรงกับความละเอียดตอนเทรน")


def verify_live(eve_path: str, limit: int = 5000):
    """ตรวจ train/serve skew จาก eve.json จริง ก่อนเสียเวลา --train

    สร้าง feature row จาก flow event จริงในไฟล์ แล้วรายงานทีละ feature
    ว่า "มีค่าเปลี่ยนแปลงจริงไหม" — feature ที่เป็นค่าคงที่ทุก flow คือ
    feature ตาย (dead feature) ซึ่งตอนเทรนอาจมีความหมายเต็มที่ แต่ตอน
    inference ไม่มีข้อมูลจริงเลย = train/serve skew แบบที่ทำให้ระบบนี้
    ตรวจจับอะไรไม่ได้มาตลอด ตรวจตรงนี้ใช้เวลาไม่กี่วินาที ดีกว่าเทรน
    หลายชั่วโมงแล้วมารู้ทีหลัง
    """
    nids = object.__new__(HybridNIDS)  # ไม่ต้องโหลดโมเดล แค่ใช้ตัวสร้าง feature
    print("=" * 68)
    print(f"verify-live: อ่าน flow event จาก {eve_path} (สูงสุด {limit} events)")
    print("=" * 68)

    rows = []
    n_tcp_proto = 0
    n_tcp_obj = 0
    with open(eve_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(rows) >= limit:
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
            if raw.get("proto") == "TCP":
                n_tcp_proto += 1
                if raw.get("tcp"):
                    n_tcp_obj += 1
            rows.append(nids._build_feature_row(raw))

    if not rows:
        print("ไม่พบ flow event เลยในไฟล์นี้ — ตรวจ Suricata eve-log config "
              "ว่าเปิด types: flow ไว้หรือยัง")
        return

    df = pd.DataFrame(rows)
    print(f"\nflow events ที่ใช้ตรวจ: {len(df)}")
    print(f"TCP flow: {n_tcp_proto} | ในนั้นมี tcp object: {n_tcp_obj}")
    if n_tcp_proto and n_tcp_obj == 0:
        print("  ⚠️  Suricata ไม่ส่ง tcp object มากับ flow event เลย")
        print("      -> ตั้ง USE_TCP_FEATURES = False แล้วค่อย --train")
        print("      (PortScan ยังตรวจได้ผ่านชั้น temporal fan-out)")
    elif n_tcp_obj:
        print("  ✅ มี tcp object ใช้ได้ -> USE_TCP_FEATURES = True ถูกต้องแล้ว")

    print(f"\nUSE_TCP_FEATURES = {USE_TCP_FEATURES}  "
          f"(FEATURES ทั้งหมด {len(FEATURES)} ตัว)\n")
    print(f"{'feature':28s} {'distinct':>8s} {'min':>12s} {'max':>12s}  สถานะ")
    print("-" * 78)
    dead = []
    for col in FEATURES:
        if col not in df.columns:
            print(f"{col:28s} {'-':>8s} {'-':>12s} {'-':>12s}  ❌ โค้ดไม่ได้สร้าง!")
            dead.append(col)
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nuniq = int(s.nunique(dropna=True))
        status = "✅"
        if nuniq <= 1:
            status = "❌ ค่าคงที่ (dead feature)"
            dead.append(col)
        elif nuniq == 2:
            status = "ok (binary)"
        print(f"{col:28s} {nuniq:8d} {s.min():12.4g} {s.max():12.4g}  {status}")

    print("-" * 78)
    if dead:
        print(f"\n❌ พบ feature ที่ไม่มีข้อมูลจริงตอน live {len(dead)} ตัว:")
        for c in dead:
            print(f"     - {c}")
        print("\n   วิธีอ่านผล (สำคัญ — อย่าเพิ่งตกใจว่าพังทั้งหมด):")
        print("   1) http_* เป็นค่าคงที่ = ปกติ ไม่ต้องแก้")
        print("      flow event ไม่มีข้อมูล HTTP อยู่แล้ว feature กลุ่มนี้มีค่า")
        print("      จริงตอนประมวลผล http event เท่านั้น ซึ่งไม่ได้ตรวจตรงนี้")
        print("   2) tcp_* เป็นค่าคงที่ 'ทั้งกลุ่ม' = ปัญหาจริง")
        print("      แปลว่า Suricata ไม่ส่ง tcp object มา -> ตั้ง")
        print("      USE_TCP_FEATURES = False แล้วค่อย --train")
        print("      (PortScan ยังตรวจได้ผ่านชั้น temporal fan-out อยู่ดี)")
        print("   3) ตัวอื่นที่คงที่ทีละตัว = อาจแค่ 'ตัวอย่างน้อย/ไม่หลากหลาย'")
        print("      ไม่ใช่ feature ตายจริง เช่นถ้าเก็บ eve.json ตอนที่มีแต่")
        print("      traffic แบบเดียว ค่าก็ย่อมไม่เปลี่ยน — ลองเก็บ eve.json")
        print("      ช่วงที่มี traffic หลากหลายกว่านี้แล้วรันซ้ำก่อนสรุป")
        print("      (จำนวน event ที่ใช้ตรวจรอบนี้ = "
              f"{len(df)} ถ้าน้อยกว่าหลักพันให้ระวังข้อนี้เป็นพิเศษ)")
    else:
        print("\n✅ ทุก feature มีค่าเปลี่ยนแปลงจริงตอน live — พร้อม --train")
    print()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", metavar="PATH")
    group.add_argument("--verify-live", metavar="EVE_JSON", dest="verify_live",
                        help="ตรวจ train/serve skew จาก eve.json จริง "
                             "(รันก่อน --train เสมอ ใช้เวลาไม่กี่วินาที)")
    group.add_argument("--realtime", metavar="PATH")
    parser.add_argument("--model_dir", default="./model")
    parser.add_argument("--workers", type=int, default=None,
                         help="จำนวน worker thread สำหรับ --realtime "
                              "(default: min(8, จำนวน CPU core))")
    parser.add_argument("--xgb_only", action="store_true",
                         help="โหมดทดสอบเร็ว: ใช้เฉพาะ XGBoost ต่อ task "
                              "(ข้าม DT/RF) ไม่ต้อง retrain — threshold "
                              "ที่ tune ไว้อาจไม่ optimal เป๊ะ ใช้ชั่วคราว "
                              "ระหว่างพัฒนาเท่านั้น")
    args = parser.parse_args()

    if args.verify_live:
        verify_live(args.verify_live)
    elif args.train:
        train_models(args.train, args.model_dir)
    elif args.realtime:
        # เดิมบรรทัดนี้ mkdir แบบไม่มี fallback เลย ถ้ารันไม่ sudo แล้ว
        # /var/log เขียนไม่ได้ จะ crash ตรงนี้ทันทีตั้งแต่ก่อนเข้า
        # HybridNIDS(...) เลย ไม่ทันได้ไปถึง PermissionError fallback ที่
        # handle_alert() มี — ห่อ try/except ให้เหมือนกันเพื่อรันแบบไม่
        # sudo ได้จริง
        try:
            Path(ML_LOG_DIR).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass  # handle_alert() จะ fallback ไปเขียน ./sword_detection_logs เองตอนมี alert จริง
        HybridNIDS(model_dir=args.model_dir, xgb_only=args.xgb_only).monitor_realtime(
            args.realtime, num_workers=args.workers)


if __name__ == "__main__":
    main()