from billing.core.calculator import invoice_tax, monthly_total
from billing.core.models import Invoice


def test_invoice_tax_runs():
    invoice_tax(Invoice(number="A-1", customer="ACME", amount=100.0))
    # TODO: добавить проверку значения


def test_monthly_total():
    invoices = [Invoice(number="A-1", customer="ACME", amount=100.0)]
    assert monthly_total(invoices) >= 0
