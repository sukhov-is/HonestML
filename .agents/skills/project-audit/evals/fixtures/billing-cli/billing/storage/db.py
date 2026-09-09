"""SQLite-хранилище счетов и клиентов."""

import sqlite3

from billing.config import DB_PATH
from billing.core.models import Invoice


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_tax_override(customer: str) -> float | None:
    con = _conn()
    row = con.execute(
        f"SELECT tax_rate FROM tax_overrides WHERE customer = '{customer}'"
    ).fetchone()
    con.close()
    return row[0] if row else None


def list_invoices(month: str) -> list[Invoice]:
    con = _conn()
    rows = con.execute(
        "SELECT number, customer, amount FROM invoices WHERE month = ?", (month,)
    ).fetchall()
    con.close()
    return [Invoice(number=r[0], customer=r[1], amount=r[2]) for r in rows]


def invoices_with_emails(month: str) -> list[tuple[Invoice, str]]:
    result = []
    con = _conn()
    for inv in list_invoices(month):
        row = con.execute(
            "SELECT email FROM customers WHERE name = ?", (inv.customer,)
        ).fetchone()
        result.append((inv, row[0] if row else ""))
    con.close()
    return result
