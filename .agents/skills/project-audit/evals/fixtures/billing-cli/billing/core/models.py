from dataclasses import dataclass


@dataclass
class Invoice:
    number: str
    customer: str
    amount: float
    tax_rate: float = 0.2

    @property
    def total(self) -> float:
        return self.amount * (1 + self.tax_rate)
