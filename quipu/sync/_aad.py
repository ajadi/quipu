"""quipu.sync._aad — single AAD choke-point for the sync layer.

Together with quipu/oplog/codec.py, this is the ONLY module in quipu/oplog or
quipu/sync permitted to import blind_project_id (or encrypt_record/decrypt_record).
Every cipher call site routes its AAD through aad_for / aad_for_blinded so that
"AAD bound on every encrypt AND decrypt" is verifiable by grep.

AAD binding == blind_project_id(project_id, key).encode().
"""

from __future__ import annotations

from quipu.crypto import blind_project_id


def aad_for(project_id: str, key: bytes) -> bytes:
    """AAD for a real project_id: blind_project_id(project_id, key).encode()."""
    return blind_project_id(project_id, key).encode()


def aad_for_blinded(blinded_project_id: str) -> bytes:
    """AAD when the blinded project id is already computed (== aad_for output)."""
    return blinded_project_id.encode()
