"""Месячный отчёт по счетам."""

from billing.core.calculator import invoice_tax
from billing.core.models import Invoice


def format_lines(invoices: list[Invoice]) -> list[str]:
    lines = []
    for inv in invoices:
        amount = f"{inv.amount:>10.2f}"
        tax = f"{invoice_tax(inv):>8.2f}"
        customer = inv.customer[:20].ljust(20)
        marker = "*" if inv.amount > 10_000 else " "
        lines.append(f"{marker} {inv.number:<8} {customer} {amount} {tax}")
    return lines


def render(invoices: list[Invoice], month: str) -> str:
    body = "\n".join(format_lines(invoices))
    return f"Отчёт за {month}\n{body}\n"
