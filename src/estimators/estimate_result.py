from dataclasses import dataclass


@dataclass
class EstimateResult:
    value: float = -1
    success: bool = False
    message: str = "No message"
