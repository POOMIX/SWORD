"""
utils/flow_finalizer.py — Session Finalization + ML Classification
ปิด Statistical และ Behavioral analysis
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger("hybrid-nids")

# ── Threshold ─────────────────────────────────────────
ZERO_BYTE_THRESHOLD  = 3      # จำนวน zero-byte flow ขั้นต่ำ
ZERO_BYTE_RATE_LIMIT = 1.0    # flow/sec ขั้นต่ำ ถึงจะ flag


@dataclass
class ZeroByteTracker:
    count: int   = 0
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)


class FlowFinalizer:
    """
    รับ SuricataSession ที่ finalize แล้ว:
    1. ดึง features ด้วย AdaptiveFlowFeatureExtractor
    2. ส่งเข้า AnomalyDetector (ML only)
    3. ตรวจ zero-byte pattern (brute-force indicator เบื้องต้น)
    4. คืน result dict พร้อม predicted_class

    use_statistical และ use_behavioral ถูกปิดไว้ก่อน
    """

    def __init__(
        self,
        anomaly_detector,
        feature_extractor=None,
        use_statistical: bool = False,
        use_behavioral:  bool = False,
    ):
        self.detector        = anomaly_detector
        self.extractor       = feature_extractor  # อาจ inject จากภายนอก
        self.use_statistical = use_statistical
        self.use_behavioral  = use_behavioral

        # zero-byte tracker: {src_ip: {dest_key: ZeroByteTracker}}
        self._zero_byte: dict = defaultdict(lambda: defaultdict(ZeroByteTracker))

    # ─────────────────────────────────────────────────────
    #  Public
    # ─────────────────────────────────────────────────────

    def process_session(self, session, features_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Parameters
        ----------
        session     : SuricataSession dataclass
        features_df : DataFrame จาก AdaptiveFlowFeatureExtractor (optional)
                      ถ้าไม่ส่งมา finalizer จะ extract เอง (ต้อง inject extractor)

        Returns
        -------
        dict ที่มี field ครบสำหรับ handle_alert()
        """
        # ── 1. Extract features ──────────────────────────
        if features_df is None:
            if self.extractor is not None:
                features_df = self.extractor.extract_from_flow(session)
            else:
                features_df = self._session_to_df(session)

        if features_df is None or features_df.empty:
            return self._empty_result(session)

        # ── 2. Check zero-byte pattern ───────────────────
        zero_byte_info = self._check_zero_byte(session)

        # ── 3. ML Detection ──────────────────────────────
        detection = self.detector.detect_anomalies(features_df, session=session)

        # ── 4. Merge results ─────────────────────────────
        result = self._build_result(session, detection, zero_byte_info)
        return result

    # ─────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────

    def _check_zero_byte(self, session) -> dict:
        """
        ตรวจ zero-byte flow pattern เพื่อช่วยจับ brute-force เบื้องต้น
        (ใช้ rule-based เป็น hint เสริม ML — ไม่ override class)
        """
        total_bytes = (
            getattr(session, "total_fwd_bytes", 0) +
            getattr(session, "total_bwd_bytes", 0)
        )
        if total_bytes > 0:
            return {"detected": False}

        src_ip   = getattr(session, "saddr", "")
        dst_ip   = getattr(session, "daddr", "")
        dst_port = getattr(session, "dport", "")
        proto    = getattr(session, "proto", "")
        now      = time.time()

        dest_key = f"{dst_ip}:{dst_port}:{proto}"
        tracker  = self._zero_byte[src_ip][dest_key]
        tracker.count     += 1
        tracker.last_seen  = now

        if tracker.count >= ZERO_BYTE_THRESHOLD:
            elapsed = max(tracker.last_seen - tracker.first_seen, 0.001)
            rate    = tracker.count / elapsed
            if rate > ZERO_BYTE_RATE_LIMIT:
                return {
                    "detected":  True,
                    "rate":      rate,
                    "count":     tracker.count,
                    "dest_key":  dest_key,
                }

        return {"detected": False}

    def _build_result(self, session, detection: dict, zero_byte_info: dict) -> dict:
        """รวม ML result + zero-byte hint → final result dict"""
        is_attack       = detection["is_attack"]
        predicted_class = detection["predicted_class"]
        confidence      = detection["confidence"]

        # ถ้า zero-byte pattern เจอ แต่ ML บอก Benign → เพิ่ม hint แต่ไม่ override
        notes = []
        if zero_byte_info.get("detected"):
            notes.append(
                f"zero-byte pattern detected "
                f"(rate={zero_byte_info.get('rate', 0):.1f}/s)"
            )

        return {
            "flow_id":         getattr(session, "flow_id", ""),
            "timestamp":       getattr(session, "starttime", time.time()),
            "src_ip":          getattr(session, "saddr", ""),
            "dst_ip":          getattr(session, "daddr", ""),
            "dst_port":        getattr(session, "dport", ""),
            "proto":           getattr(session, "proto", ""),
            "app_proto":       getattr(session, "appproto", ""),
            "duration":        getattr(session, "duration", 0),
            "total_bytes":     (getattr(session, "total_fwd_bytes", 0) +
                                getattr(session, "total_bwd_bytes", 0)),
            "total_packets":   (getattr(session, "total_fwd_packets", 0) +
                                getattr(session, "total_bwd_packets", 0)),
            # ML result
            "is_attack":       is_attack,
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "proba_per_class": detection.get("proba_per_class", {}),
            "model_votes":     detection.get("model_votes", {}),
            "ml_result":       detection.get("ml_result", {}),
            # hints
            "zero_byte_info":  zero_byte_info,
            "notes":           notes,
        }

    def _empty_result(self, session) -> dict:
        return {
            "flow_id":         getattr(session, "flow_id", ""),
            "timestamp":       getattr(session, "starttime", time.time()),
            "src_ip":          getattr(session, "saddr", ""),
            "dst_ip":          getattr(session, "daddr", ""),
            "dst_port":        getattr(session, "dport", ""),
            "proto":           getattr(session, "proto", ""),
            "is_attack":       False,
            "predicted_class": "Unknown",
            "confidence":      0.0,
            "proba_per_class": {},
            "model_votes":     {},
            "notes":           ["feature extraction failed"],
        }

    def _session_to_df(self, session) -> Optional[pd.DataFrame]:
        """
        Fallback: แปลง session เป็น DataFrame ด้วย field ที่รู้จัก
        ใช้เมื่อไม่มี AdaptiveFlowFeatureExtractor inject มา
        """
        try:
            dur = getattr(session, "duration", 0) or 1e-6
            fwd_bytes = getattr(session, "total_fwd_bytes", 0)
            bwd_bytes = getattr(session, "total_bwd_bytes", 0)
            fwd_pkts  = getattr(session, "total_fwd_packets", 0)
            bwd_pkts  = getattr(session, "total_bwd_packets", 0)
            total_bytes = fwd_bytes + bwd_bytes
            total_pkts  = fwd_pkts + bwd_pkts

            row = {
                "dest_port":            int(getattr(session, "dport", 0) or 0),
                "duration":             float(dur),
                "total_fwd_packets":    int(fwd_pkts),
                "total_bwd_packets":    int(bwd_pkts),
                "total_fwd_bytes":      int(fwd_bytes),
                "total_bwd_bytes":      int(bwd_bytes),
                "flow_bytes_per_sec":   total_bytes / dur,
                "flow_packets_per_sec": total_pkts  / dur,
                "down_up_ratio":        bwd_bytes / max(fwd_bytes, 1),
            }
            return pd.DataFrame([row])
        except Exception as e:
            logger.debug(f"_session_to_df error: {e}")
            return None
