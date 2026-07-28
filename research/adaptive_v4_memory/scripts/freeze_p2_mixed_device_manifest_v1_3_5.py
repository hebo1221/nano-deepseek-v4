from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))

import p2_direct_attestation as attestation  # noqa: E402
import p2_direct_controller_contract_v1_3_5 as contract  # noqa: E402
import probe_p2_direct_controller_topology_v1_3_5 as probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze one mixed-device site manifest.")
    parser.add_argument("--topology-probe", type=Path, required=True)
    parser.add_argument("--attestation-key-path", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree_digest = contract.v1_3_5_implementation_tree_digest()
    if contract.v1_3_5_implementation_tree_digest_at_commit(head) != tree_digest:
        raise RuntimeError("HEAD does not reproduce the staged mixed-device implementation.")
    trust_root = attestation.load_trust_root(
        args.attestation_key_path,
        repository_root=REPOSITORY_ROOT,
    )
    probe_binding = probe.load_probe_binding(args.topology_probe, trust_root=trust_root)
    payload = contract.build_v1_3_5_manifest_payload(
        attestation_key_id=trust_root.key_id,
        implementation_tree_digest=tree_digest,
        implementation_source_commit=head,
        selected_worker_count=3,
        topology_probe_binding=probe_binding,
    )
    contract.validate_v1_3_5_manifest_payload(payload, verify_implementation=True)
    target = REPOSITORY_ROOT / contract.V1_3_5_MANIFEST_PATH
    encoded = contract.base.canonical_pretty_manifest_bytes(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o644)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
