"""Экспорт счетов."""

from abc import ABC, abstractmethod

from billing.core.calculator import invoice_tax
from billing.core.models import Invoice


class Exporter(ABC):
    @abstractmethod
    def export(self, invoices: list[Invoice]) -> str: ...


class ExporterFactory:
    """Фабрика экспортёров под будущие форматы."""

    _registry: dict[str, type[Exporter]] = {}

    @classmethod
    def register(cls, name: str):
        def deco(klass: type[Exporter]) -> type[Exporter]:
            cls._registry[name] = klass
            return klass

        return deco

    @classmethod
    def create(cls, name: str) -> Exporter:
        return cls._registry[name]()


@ExporterFactory.register("csv")
class CsvExporter(Exporter):
    def export(self, invoices: list[Invoice]) -> str:
        lines = ["number,customer,amount,tax"]
        for inv in invoices:
            amount = f"{inv.amount:>10.2f}"
            tax = f"{invoice_tax(inv):>8.2f}"
            customer = inv.customer[:20].ljust(20)
            marker = "*" if inv.amount > 10_000 else " "
            lines.append(f"{marker} {inv.number:<8} {customer} {amount} {tax}")
        return "\n".join(lines)
