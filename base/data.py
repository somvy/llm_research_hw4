from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Data:
    samples: List[Dict[str, Any]]
    intervention: Dict[str, Any]
    difficulty: int
    seed: int
