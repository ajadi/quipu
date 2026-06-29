"""quipu.oplog — encrypted append-only operation log primitives.

Public API:
    from quipu.oplog import (
        OplogEntry,
        encode_entry, decode_entry, frame_blobs, unframe_blobs,
    )
"""

from quipu.oplog.entry import OplogEntry
from quipu.oplog.codec import decode_entry, encode_entry, frame_blobs, unframe_blobs

__all__ = [
    "OplogEntry",
    "encode_entry",
    "decode_entry",
    "frame_blobs",
    "unframe_blobs",
]
