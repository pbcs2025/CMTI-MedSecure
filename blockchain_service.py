"""
Blockchain helper utilities for MedSecure.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

try:
    from web3 import Web3  # type: ignore
except ImportError:
    Web3 = None


GANACHE_URL = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")

# Fallback ABI for MedSecure.sol so writes can still work if
# blockchain/MedSecure.abi.json is not present.
DEFAULT_CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "imageId", "type": "string"},
            {"internalType": "string", "name": "dataHash", "type": "string"},
            {"internalType": "string", "name": "riskLevel", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "storeRecord",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "imageId", "type": "string"}],
        "name": "getRecords",
        "outputs": [
            {
                "components": [
                    {"internalType": "string", "name": "imageId", "type": "string"},
                    {"internalType": "string", "name": "dataHash", "type": "string"},
                    {"internalType": "string", "name": "riskLevel", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                ],
                "internalType": "struct MedSecure.Record[]",
                "name": "",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


def _default_risk(prediction: str) -> str:
    prediction_norm = str(prediction).strip().lower()
    if prediction_norm == "genuine":
        return "LOW"
    return "HIGH"


def confidence_to_percent(confidence: float) -> float:
    """Normalize API confidence to a 0–100 percentage for hashing and display."""
    c = float(confidence)
    if 0.0 <= c <= 1.0:
        return round(c * 100.0, 1)
    return round(c, 1)


def normalize_risk_for_chain(risk: str) -> str:
    """Map model/API risk strings to uppercase labels stored on-chain."""
    r = str(risk).strip().lower()
    if r in ("low",):
        return "LOW"
    if r in ("moderate", "medium"):
        return "MODERATE"
    if r in ("high",):
        return "HIGH"
    if r in ("unverified", "unknown"):
        return "UNVERIFIED"
    ru = str(risk).strip().upper()
    if ru in ("LOW", "MODERATE", "HIGH", "UNVERIFIED"):
        return ru
    return "HIGH"


def create_verification_record(
    *,
    record_id: Optional[str] = None,
    mode: str,
    label: str,
    status: str,
    confidence: float,
    risk_level: str,
    timestamp: Optional[int] = None,
    barcode: Optional[str] = None,
    barcode_status: Optional[str] = None,
    image_status: Optional[str] = None,
) -> dict:
    """
    Canonical payload for SHA-256 verification and MedSecure.storeRecord.
    image_only: mode == \"image_only\"
    Barcode + image: mode == \"barcode_plus_image\" (includes barcode fields in the hash).
    """
    ts = int(timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp())
    rid = record_id or str(uuid.uuid4())
    rec = {
        "record_id": rid,
        "mode": mode,
        "label": str(label),
        "status": str(status),
        "confidence_percent": confidence_to_percent(confidence),
        "risk_level": normalize_risk_for_chain(risk_level),
        "timestamp": ts,
    }
    if mode == "barcode_plus_image":
        rec["barcode"] = str(barcode or "")
        rec["barcode_status"] = str(barcode_status or "")
        if image_status is not None:
            rec["image_status"] = str(image_status)
    return rec


def create_result_object(prediction: str, confidence: float, risk_level: Optional[str] = None) -> dict:
    """
    Build an image-only verification record for /store-result (legacy API).
    """
    rl = risk_level or _default_risk(prediction)
    st = "genuine" if str(prediction).strip().lower() == "genuine" else "suspicious"
    return create_verification_record(
        mode="image_only",
        label=str(prediction),
        status=st,
        confidence=confidence,
        risk_level=rl,
    )


def blockchain_configured() -> bool:
    return bool(os.getenv("MEDSECURE_CONTRACT_ADDRESS", "").strip()) and Web3 is not None


def try_commit_to_chain(record: dict) -> dict:
    """
    Anchor verification metadata on Ganache when configured; never raises.
    Returns fields suitable for JSON responses alongside scan results.
    """
    verification_hash = generate_verification_hash(record)
    rid = record["record_id"]
    risk = record["risk_level"]
    ts = record["timestamp"]

    thanks_stored = (
        "Thank you — your scan is now part of the MediSecure verification ledger. "
        "Every contribution helps protect the community."
    )
    thanks_pending = (
        "Thank you for using MediScan and helping build a safer medicine ecosystem. "
        "When the MediSecure node is online, your scan fingerprints will anchor on-chain like others."
    )

    if not blockchain_configured():
        return {
            "blockchain_stored": False,
            "verification_hash": verification_hash,
            "record_id": rid,
            "blockchain_message": (
                "A cryptographic fingerprint of this result was prepared. "
                "The server is not connected to a MediSecure chain right now, so nothing was written on-chain."
            ),
            "contributor_thanks": thanks_pending,
        }

    try:
        tx_hash = store_on_chain(rid, verification_hash, risk, ts)
        return {
            "blockchain_stored": True,
            "blockchain_tx_hash": tx_hash,
            "verification_hash": verification_hash,
            "record_id": rid,
            "blockchain_message": (
                "Success: a tamper-evident fingerprint of this scan is recorded on the MediSecure blockchain."
            ),
            "contributor_thanks": thanks_stored,
        }
    except Exception as e:
        return {
            "blockchain_stored": False,
            "verification_hash": verification_hash,
            "record_id": rid,
            "blockchain_message": f"Could not write to the blockchain: {e}",
            "contributor_thanks": thanks_pending,
        }


def generate_verification_hash(result_obj: dict) -> str:
    """
    SHA-256 of canonical JSON (sorted keys).
    """
    canonical = json.dumps(result_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_contract(web3: Web3):
    contract_address = os.getenv("MEDSECURE_CONTRACT_ADDRESS", "").strip()
    abi_path = os.getenv(
        "MEDSECURE_CONTRACT_ABI_PATH",
        os.path.join("blockchain", "MedSecure.abi.json"),
    )

    if not contract_address:
        raise ValueError("Set MEDSECURE_CONTRACT_ADDRESS environment variable.")

    if os.path.exists(abi_path):
        with open(abi_path, "r", encoding="utf-8") as f:
            abi = json.load(f)
    else:
        abi = DEFAULT_CONTRACT_ABI

    return web3.eth.contract(
        address=web3.to_checksum_address(contract_address),
        abi=abi,
    )


def store_on_chain(image_id: str, hash_val: str, risk: str, timestamp: int) -> str:
    """
    Store verification metadata on Ganache and return transaction hash.
    """
    if Web3 is None:
        raise ImportError(
            "web3 is not installed. Install dependencies with: pip install -r requirements.txt"
        )
    web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not web3.is_connected():
        raise ConnectionError(f"Could not connect to Ganache at {GANACHE_URL}")

    contract = _load_contract(web3)
    accounts = web3.eth.accounts
    if not accounts:
        raise RuntimeError("No accounts found in Ganache.")

    tx_hash = contract.functions.storeRecord(
        image_id,
        hash_val,
        risk,
        int(timestamp),
    ).transact({"from": accounts[0]})

    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.transactionHash.hex()