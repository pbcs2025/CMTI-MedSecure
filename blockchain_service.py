"""
Blockchain helper utilities for MedSecure.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

from web3 import Web3


GANACHE_URL = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("MEDSECURE_CONTRACT_ADDRESS", "")
CONTRACT_ABI_PATH = os.getenv(
    "MEDSECURE_CONTRACT_ABI_PATH",
    os.path.join("blockchain", "MedSecure.abi.json"),
)


def _default_risk(prediction: str) -> str:
    prediction_norm = str(prediction).strip().lower()
    if prediction_norm == "genuine":
        return "LOW"
    return "HIGH"


def create_result_object(prediction: str, confidence: float, risk_level: str | None = None) -> dict:
    """
    Build canonical result object:
    {
      image_id, prediction, confidence, risk_level, timestamp
    }
    """
    return {
        "image_id": str(uuid.uuid4()),
        "prediction": prediction,
        "confidence": float(confidence),
        "risk_level": risk_level or _default_risk(prediction),
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
    }


def generate_verification_hash(result_obj: dict) -> str:
    """
    SHA-256 of canonical JSON (sorted keys).
    """
    canonical = json.dumps(result_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_contract(web3: Web3):
    if not CONTRACT_ADDRESS:
        raise ValueError("Set MEDSECURE_CONTRACT_ADDRESS environment variable.")
    if not os.path.exists(CONTRACT_ABI_PATH):
        raise FileNotFoundError(f"Contract ABI file not found: {CONTRACT_ABI_PATH}")

    with open(CONTRACT_ABI_PATH, "r", encoding="utf-8") as f:
        abi = json.load(f)

    return web3.eth.contract(
        address=web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=abi,
    )


def store_on_chain(image_id: str, hash_val: str, risk: str, timestamp: int) -> str:
    """
    Store verification metadata on Ganache and return transaction hash.
    """
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
