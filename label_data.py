#   CADETS E3 ground truth mapping
#
#   Ground truth extracted from:
#     TC_Ground_Truth_Report_E3_Update.pdf  (Kudu Dynamics / DARPA TC)
#
#   CADETS attack days (April 2018)
#   ─────────────────────────────────────────────────────────────────
#   Day  6  §3.1   Nginx backdoor, Drakon in-memory, netrecon, sshd inject
#                   → CADETS crash (kernel panic)
#   Day  6  §4.1   Common-threat phishing e-mail sent *via* CADETS postfix
#   Day 10  §4.1   Phishing e-mail again routed through CADETS postfix
#   Day 11  §3.8   Nginx backdoor, Drakon in-memory, grain, sshd inject
#                   → CADETS crash again
#   Day 12  §3.13  Nginx backdoor, Drakon in-memory, tmux-1002, minions,
#                   font, XIM, sendmail, test, micro APT port scans
#   Day 13  §3.14  Nginx backdoor, Drakon, pEja72mA, eWq10bVcx,
#                   eraseme, memhelp.so, done.so, sshd inject
#
#   BENIGN days for CADETS:  2, 3, 4, 5, 7, 8, 9
#   ATTACK days for CADETS:  6, 10, 11, 12, 13
#   ─────────────────────────────────────────────────────────────────

from typing import Set, Dict, Any, Optional

MALICIOUS_IPS_BY_DAY: Dict[int, Set[str]] = {
    6: {
        # §3.1 Nation State attack
        "81.49.200.166",    # http_post exploit
        "78.205.235.65",    # shellcode_server
        "200.36.109.214",   # loaderDrakon
        "139.123.0.113",    # drakon
        "152.111.159.139",  # libdrakon
        "154.143.113.18",   # netrecon (failed)
        "61.167.39.128",    # netrecon (success)
        # §4.1 Common Threat phishing via CADETS e-mail server
        "62.83.155.175",    # phishing attack
    },
    10: {
        # §4.1.2.3  phishing e-mail routed through CADETS postfix
        "62.83.155.175",    # phishing attack
    },
    11: {
        # §3.8  Nginx backdoor w/ Drakon
        "25.159.96.207",    # http_post
        "76.56.184.25",     # shellcode_server
        "155.162.39.48",    # loaderDrakon
        "198.115.236.119",  # libdrakon (failed)
    },
    12: {
        # §3.13  Nginx backdoor w/ Drakon + Micro APT
        "25.159.96.207",    # webserver
        "76.56.184.25",     # shellcode_server
        "155.162.39.48",    # loaderDrakon
        "198.115.236.119",  # libdrakon (failed)
        "53.158.101.118",   # drakon
        "98.15.44.232",     # micro (failed)
        "192.113.144.28",   # micro 2 (sendmail)
    },
    13: {
        # §3.14  Nginx backdoor w/ Drakon
        "25.159.96.207",    # webserver
        "76.56.184.25",     # shellcode_server
        "155.162.39.48",    # loaderDrakon
        "198.115.236.119",  # libdrakon
        "53.158.101.118",   # drakon
    },
}

MALICIOUS_FILES_BY_DAY: Dict[int, Set[str]] = {
    6: {
        # §3.1
        "/tmp/vUgefal",
        "/var/log/devc",
    },
    11: {
        # §3.8
        "/tmp/grain",
    },
    12: {
        # §3.13
        "/tmp/tmux-1002",
        "/tmp/minions",
        "/tmp/font",
        "/tmp/XIM",
        "/tmp/test",
        "/var/log/netlog",
        "/var/log/sendmail",
        "/tmp/main",
    },
    13: {
        # §3.14
        "/tmp/pEja72mA",
        "/tmp/eWq10bVcx",
        "/tmp/memhelp.so",
        "/tmp/eraseme",
        "/tmp/done.so",
    },
}

MALICIOUS_PROCESSES_BY_DAY: Dict[int, Set[str]] = {
    6: {
        "vUgefal",      # elevated drakon process
    },
    12: {
        "XIM",          # drakon implant process
        "test",         # micro apt executed as /tmp/test
    },
    13: {
        "pEja72mA",     # drakon implant executable
    },
}

ALL_MALICIOUS_IPS: Set[str] = set()
for _ips in MALICIOUS_IPS_BY_DAY.values():
    ALL_MALICIOUS_IPS |= _ips

ALL_MALICIOUS_FILES: Set[str] = set()
for _fps in MALICIOUS_FILES_BY_DAY.values():
    ALL_MALICIOUS_FILES |= _fps

ALL_MALICIOUS_PROCESSES: Set[str] = set()
for _ps in MALICIOUS_PROCESSES_BY_DAY.values():
    ALL_MALICIOUS_PROCESSES |= _ps

# The set of days on which *any* CADETS attack occurred
ATTACK_DAYS: Set[int] = {6, 10, 11, 12, 13}

# Relation types
EXEC_RELATIONS  = {"EVENT_EXECUTE", "EVENT_EXECVE", "EVENT_FORK", "EVENT_CLONE"}
WRITE_RELATIONS = {"EVENT_WRITE", "EVENT_RENAME", "EVENT_TRUNCATE",
                   "EVENT_MODIFY_FILE_ATTRIBUTES"}
NETWORK_RELATIONS = {"EVENT_SENDTO", "EVENT_CONNECT", "EVENT_SENDMSG"}
READ_RELATIONS  = {"EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"}

SYSTEM_PROCESSES = {
    "systemd", "sshd", "init", "kernel", "kthreadd", "kworker",
    "ksoftirqd", "migration", "rcu", "watchdog",
}


def _match_ip(properties: str, ip_set: Set[str]) -> bool:
    """Check if *properties* contains any IP from *ip_set*."""
    if not properties:
        return False
    for ip in ip_set:
        if ip in properties:
            return True
    return False


def _match_file(properties: str, file_set: Set[str]) -> bool:
    """Exact-path matching.

    We require that the file path either equals the candidate or that the
    candidate is a suffix that starts at a path boundary.  This avoids
    false positives like '/usr/.../salt/module' matching bare 'done.so'.
    """
    if not properties:
        return False
    prop = properties.strip()
    for path in file_set:
        if prop == path:
            return True
        # Also match if prop ends with the path after a separator
        if prop.endswith(path) and len(prop) > len(path):
            preceding = prop[-(len(path) + 1)]
            if preceding in ('/', '\\', ' '):
                return True
    return False


def _match_process(properties: str, proc_set: Set[str]) -> bool:
    """Exact process-name matching (basename)."""
    if not properties:
        return False
    prop = properties.strip()
    basename = prop.rsplit("/", 1)[-1]
    for pname in proc_set:
        if basename == pname or prop == pname:
            return True
    return False

def detect_anchor(node_type: str, properties: str,
                  day: Optional[int] = None) -> tuple:
    """Return (is_anchor, reason) for a node, scoped to *day*.

    If *day* is not in ATTACK_DAYS the answer is always (False, '').
    """
    if day is not None and day not in ATTACK_DAYS:
        return False, ""

    if node_type == "netflow":
        ip_set = MALICIOUS_IPS_BY_DAY.get(day, set()) if day else ALL_MALICIOUS_IPS
        if _match_ip(properties, ip_set):
            return True, f"malicious_ip_day{day}"

    elif node_type == "file":
        file_set = MALICIOUS_FILES_BY_DAY.get(day, set()) if day else ALL_MALICIOUS_FILES
        if _match_file(properties, file_set):
            return True, f"malicious_file_day{day}"

    elif node_type == "subject":
        proc_set = MALICIOUS_PROCESSES_BY_DAY.get(day, set()) if day else ALL_MALICIOUS_PROCESSES
        if _match_process(properties, proc_set):
            return True, f"malicious_process_day{day}"

    return False, ""


def is_system_process(properties: str) -> bool:
    if not properties:
        return False
    pl = properties.lower()
    return any(sp in pl for sp in SYSTEM_PROCESSES)

class TaintTracker:
    """Track malicious (tainted) nodes.

    A separate TaintTracker should be created for **each day** so that
    taint from one day does not leak into another.
    """

    def __init__(self):
        self.malicious_nodes: Set[str] = set()
        self.anchor_reasons: Dict[str, str] = {}

    def is_mal(self, h: str) -> bool:
        return h in self.malicious_nodes

    is_malicious = is_mal          # alias

    def mark_mal(self, h: str, reason: str):
        if h not in self.malicious_nodes:
            self.malicious_nodes.add(h)
            self.anchor_reasons[h] = reason

    mark_malicious = mark_mal      # alias

    def get_stats(self) -> Dict[str, Any]:
        return {
            "Total Malicious Nodes": len(self.malicious_nodes),
            "Anchor Reasons": dict(self.anchor_reasons),
        }

def should_propagate_taint(
    src_type: str, dst_type: str, relation_type: str,
    src_malicious: bool, dst_malicious: bool,
    src_properties: str, dst_properties: str,
) -> tuple:
    """Decide whether taint should flow from src to dst (or vice versa)."""

    if src_malicious and src_type == "subject" and dst_type == "subject":
        if relation_type in EXEC_RELATIONS:
            if not is_system_process(dst_properties):
                return True, "exec_from_malicious_process"

    if src_malicious and src_type == "subject" and dst_type == "file":
        if relation_type in WRITE_RELATIONS:
            return True, "write_from_malicious_process"

    if src_malicious and src_type == "subject" and dst_type == "file":
        if relation_type in EXEC_RELATIONS:
            return True, "exec_malicious_file"
    if dst_malicious and src_type == "subject" and dst_type == "file":
        if relation_type in EXEC_RELATIONS:
            return True, "exec_malicious_file"

    if src_malicious and src_type == "subject" and dst_type == "netflow":
        if relation_type in NETWORK_RELATIONS:
            return True, "network_from_malicious_process"

    return False, ""

def compute_edge_label(
    src_hash: str, src_type: str, src_properties: str,
    dst_hash: str, dst_type: str, dst_properties: str,
    relation_type: str,
    tracker: TaintTracker,
    day: Optional[int] = None,
) -> tuple:
    """Return (reason, propagation_info) for a single edge.

    *day* gates anchor detection so that benign days produce no positives.
    """
    src_mal = tracker.is_malicious(src_hash)
    dst_mal = tracker.is_malicious(dst_hash)

    # Try to discover new anchors (only on attack days)
    if not src_mal:
        is_anchor, reason = detect_anchor(src_type, src_properties, day)
        if is_anchor:
            tracker.mark_malicious(src_hash, reason)
            src_mal = True

    if not dst_mal:
        is_anchor, reason = detect_anchor(dst_type, dst_properties, day)
        if is_anchor:
            tracker.mark_malicious(dst_hash, reason)
            dst_mal = True

    if src_mal or dst_mal:
        should, prop_reason = should_propagate_taint(
            src_type, dst_type, relation_type,
            src_mal, dst_mal,
            src_properties, dst_properties,
        )
        if should:
            if not dst_mal:
                tracker.mark_malicious(dst_hash, prop_reason)
            return prop_reason, prop_reason

        # Edge touches a malicious node but propagation rules don't fire
        if src_mal:
            return tracker.anchor_reasons.get(src_hash, "malicious_src"), "involves_malicious"
        if dst_mal:
            return tracker.anchor_reasons.get(dst_hash, "malicious_dst"), "involves_malicious"

    return "Benign", ""
