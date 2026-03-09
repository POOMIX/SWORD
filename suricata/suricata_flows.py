from dataclasses import dataclass, field
from typing import Union, List, Dict


@dataclass
class SuricataFlow:
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: Union[str, bool]
    starttime: str
    endtime: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    state: str
    dur: Union[float, bool] = False      # flow duration (seconds) จาก flow.age
    pkts: Union[int, bool] = False
    bytes: Union[int, bool] = False
    smac: str = ""
    dmac: str = ""
    dir_: str = "->"
    type_: str = "conn"
    flow_source: str = "suricata"

    # ── TCP Flag fields ───────────────────────────────────────────────────
    # ดึงมาจาก tcp.syn, tcp.rst, tcp.ack, tcp.fin ใน Suricata flow event
    # ใช้แยก DoS (syn+ack สูง) vs PortScan (rst สูง) vs Benign
    tcpflags: str = ""      # hex string เช่น "1f" จาก tcp.tcp_flags
    tcp_syn: bool = False
    tcp_rst: bool = False
    tcp_ack: bool = False
    tcp_fin: bool = False

    def __post_init__(self):
        # dur: ใช้ flow.age (วินาที) ที่ส่งมาตรงๆ แทนการคำนวณจาก start/end
        # เพราะ flow.age เป็น integer วินาที ที่ Suricata คำนวณไว้แล้ว
        if not self.dur:
            self.dur = 0
        self.pkts  = self.dpkts + self.spkts
        self.bytes = self.dbytes + self.sbytes
        self.uid   = str(self.uid)


@dataclass
class SuricataHTTP:
    starttime: str
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: str
    method: str
    host: str
    uri: str
    user_agent: str
    status_code: str
    version: str
    request_body_len: int
    response_body_len: int
    status_msg: str = ""
    resp_mime_types: str = ""
    resp_fuids: str = ""
    type_: str = "http"
    flow_source: str = "suricata"


@dataclass
class SuricataDNS:
    starttime: str
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: str
    query: str
    TTLs: str
    qtype_name: str
    answers: List[Dict[str, str]]
    qclass_name: str = ""
    rcode_name: str = ""
    type_: str = "dns"
    flow_source: str = "suricata"


@dataclass
class SuricataTLS:
    starttime: str
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: str
    sslversion: str
    subject: str
    issuer: str
    server_name: str
    notbefore: str
    notafter: str
    type_: str = "ssl"
    flow_source: str = "suricata"


@dataclass
class SuricataFile:
    starttime: str
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: str
    size: int
    type_: str = "files"
    flow_source: str = "suricata"
    md5: str = ""
    sha1: str = ""
    source: str = ""
    analyzers: str = ""
    tx_hosts: str = ""
    rx_hosts: str = ""


@dataclass
class SuricataSSH:
    starttime: str
    uid: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    appproto: str
    client: str
    version: str
    server: str
    auth_success: str = ""
    auth_attempts: str = ""
    cipher_alg: str = ""
    mac_alg: str = ""
    kex_alg: str = ""
    compression_alg: str = ""
    host_key_alg: str = ""
    host_key: str = ""
    type_: str = "ssh"
    flow_source: str = "suricata"
