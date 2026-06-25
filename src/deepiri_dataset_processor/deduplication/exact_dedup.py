"""
Exact deduplication using hash-based detection.

No external dependencies required (stdlib only).
"""

import hashlib
from collections import defaultdict
from typing import Any, Dict, List


class ExactDeduplicator:
    """
    Hash-based exact duplicate detection.

    Uses MD5 hashing for fast duplicate detection. For content-aware
    deduplication, use SemanticDeduplicationEngine instead.
    """

    def find_duplicates(self, texts: List[str]) -> Dict[str, Any]:
        """
        Find exact duplicate texts.

        Args:
            texts: List of texts to check

        Returns:
            Report with duplicate groups and statistics
        """
        text_hashes: Dict[str, int] = {}
        duplicates: Dict[str, List[int]] = defaultdict(list)

        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in text_hashes:
                duplicates[text_hash].append(i)
            else:
                text_hashes[text_hash] = i
                duplicates[text_hash] = [i]

        duplicate_groups = {
            h: indices for h, indices in duplicates.items() if len(indices) > 1
        }
        total_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())

        return {
            "duplicates_detected": len(duplicate_groups) > 0,
            "duplicate_groups": len(duplicate_groups),
            "total_duplicate_instances": total_duplicates,
            "duplicate_rate": total_duplicates / len(texts) if texts else 0.0,
            "groups": duplicate_groups,
        }

    def filter_duplicates(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts, keeping the first occurrence.

        Args:
            texts: List of texts

        Returns:
            Deduplicated list
        """
        seen: set = set()
        unique: List[str] = []

        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(text)

        return unique
