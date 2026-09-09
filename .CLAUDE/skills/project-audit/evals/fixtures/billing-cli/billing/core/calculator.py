"""Расчёт налогов и итогов по счетам."""

from billing.core.models import Invoice
from billing.storage.db import get_tax_override


def invoice_tax(invoice: Invoice) -> float:
    try:
        override = get_tax_override(invoice.customer)
        rate = override if override is not None else invoice.tax_rate
        # TODO(2024-03): убрать костыль округления после миграции тарифов
        return round(invoice.amount * rate, 2)
    except Exception:
        return 0.0


def monthly_total(invoices: list[Invoice]) -> float:
    total = 0.0
    for inv in invoices:
        total += inv.amount + invoice_tax(inv)
    return total
