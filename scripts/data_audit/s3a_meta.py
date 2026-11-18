"""S3A 输出元数据块（Gate 1.4）。

9A–9M 输出 JSON 统一通过 attach_meta(...) 注入元数据，使任一 headline 数字都能
回指到唯一 spec 与输入包：spec_id / spec_hash / input_hashes / network_version /
run_at / command / script / status。
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/s3a_final_spec_v1.yaml'
_SPEC_BYTES = SPEC.read_bytes()
_SPEC_HASH = hashlib.sha256(_SPEC_BYTES).hexdigest()


def attach_meta(payload, script_file=None, network_version=None, status='COMPUTED', **extra):
    """把统一元数据写入 payload（dict）并返回同一个对象。"""
    import yaml
    spec = yaml.safe_load(_SPEC_BYTES.decode('utf-8'))
    meta = {
        'spec_id': spec['spec_id'],
        'spec_hash': _SPEC_HASH,
        'input_hashes': spec['provenance']['input_hashes'],
        'network_version': network_version or spec['network']['primary_version'],
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'script': Path(script_file).name if script_file else None,
        'status': status,
    }
    meta.update(extra)
    payload.update(meta)
    return payload
