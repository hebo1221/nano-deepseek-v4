# Security policy

## Supported versions

Security fixes are applied to the latest tagged release and the `main` branch.
Older releases should be upgraded before reporting version-specific behavior.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
[private vulnerability reporting form](https://github.com/hebo1221/nano-deepseek-v4/security/advisories/new)
and include:

- the affected version or commit;
- a minimal reproduction or malformed artifact, when safe to share;
- expected impact and the trust boundary crossed;
- any suggested mitigation.

Checkpoint and cache parsers operate on local files, but callers should still
treat model artifacts as untrusted input. Prefer `safetensors`, run
`verify_deepseek_checkpoint_snapshot` before loading official snapshots, and
verify artifact digests obtained through a trusted channel.

Please allow a reasonable remediation window before public disclosure. Receipt
and remediation timelines depend on maintainer availability; no service-level
agreement is currently offered.
