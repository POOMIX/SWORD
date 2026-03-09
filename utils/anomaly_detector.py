"""
utils/anomaly_detector.py — Multiclass ML Detector
รองรับ Benign / DoS / PortScan
Statistical และ Behavioral detection ถูกปิดในเวอร์ชันนี้
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("hybrid-nids")

FEATURES = [
    "dest_port",
    "duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "down_up_ratio",
]


class AnomalyDetector:
    """
    Wrapper รับ model จาก HybridNIDS แล้ว expose ฟังก์ชัน detect_anomalies()
    สำหรับ FlowFinalizer เรียกใช้

    statistical_mode และ behavioral_mode ปิดไว้ก่อน
    """

    def __init__(self, nids_instance, use_statistical: bool = False, use_behavioral: bool = False):
        """
        Parameters
        ----------
        nids_instance : HybridNIDS
            อ้างอิงกลับไปยัง HybridNIDS เพื่อเรียก predict()
        use_statistical : bool
            ปิดไว้ก่อน (False)
        use_behavioral : bool
            ปิดไว้ก่อน (False)
        """
        self.nids           = nids_instance
        self.use_statistical = use_statistical
        self.use_behavioral  = use_behavioral

    # ─────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────

    def detect_anomalies(self, features: pd.DataFrame, session=None) -> dict:
        """
        รับ feature DataFrame คืนค่า detection result รวม

        Returns
        -------
        dict ที่มี:
            is_attack        : bool
            predicted_class  : str   ('Benign' | 'DoS' | 'PortScan')
            confidence       : float
            proba_per_class  : dict
            model_votes      : dict
            ml_result        : dict  (raw output จาก nids.predict)
        """
        ml_result = self._run_ml(features)

        return {
            "is_attack":       ml_result["is_attack"],
            "predicted_class": ml_result["predicted_class"],
            "confidence":      ml_result["confidence"],
            "proba_per_class": ml_result.get("proba_per_class", {}),
            "model_votes":     ml_result.get("model_votes", {}),
            "ml_result":       ml_result,
            # placeholder สำหรับอนาคต
            "stat_result":     {"enabled": False},
            "behavioral_result": {"enabled": False},
        }

    # ─────────────────────────────────────────────────────
    #  Internal
    # ─────────────────────────────────────────────────────

    def _run_ml(self, features: pd.DataFrame) -> dict:
        """เรียก nids.predict() และ return ผลลัพธ์"""
        try:
            return self.nids.predict(features)
        except Exception as e:
            logger.error(f"AnomalyDetector._run_ml error: {e}")
            return {
                "predicted_class": "Unknown",
                "confidence":      0.0,
                "is_attack":       False,
                "proba_per_class": {},
                "model_votes":     {},
            }

    # ─────────────────────────────────────────────────────
    #  Stubs สำหรับ compatibility (ถ้า code เก่าเรียก)
    # ─────────────────────────────────────────────────────

    def detect_ml_anomaly(self, features: pd.DataFrame) -> dict:
        """backward-compat stub"""
        return self._run_ml(features)

    def detect_statistical_anomaly(self, features: pd.DataFrame) -> dict:
        """ปิดไว้ก่อน — คืน empty result"""
        return {
            "is_anomalous": False,
            "score":        0.0,
            "enabled":      False,
        }
