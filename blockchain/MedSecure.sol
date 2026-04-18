// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract MedSecure {
    struct Record {
        string imageId;
        string dataHash;
        string riskLevel;
        uint timestamp;
    }

    mapping(string => Record[]) public records;

    function storeRecord(
        string memory imageId,
        string memory dataHash,
        string memory riskLevel,
        uint timestamp
    ) public {
        records[imageId].push(Record(imageId, dataHash, riskLevel, timestamp));
    }

    function getRecords(string memory imageId) public view returns (Record[] memory) {
        return records[imageId];
    }
}
