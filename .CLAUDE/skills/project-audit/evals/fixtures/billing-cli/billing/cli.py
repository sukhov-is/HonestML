"""Точка входа CLI."""

import argparse

from billing.report import render
from billing.storage.db import invoices_with_emails, list_invoices
from billing.storage.export import ExporterFactory


def sync_remote() -> None:
    # TODO(2024-01): дописать после выбора протокола реестра
    raise NotImplementedError("синхронизация ещё не реализована")


def main() -> None:
    parser = argparse.ArgumentParser(prog="billing")
    parser.add_argument(
        "--strict-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create")
    create.add_argument("--customer", required=True)
    create.add_argument("--amount", type=float, required=True)

    report = sub.add_parser("report")
    report.add_argument("--month", required=True)

    export = sub.add_parser("export")
    export.add_argument("--format", default="csv")
    export.add_argument("--out", required=True)

    sub.add_parser("sync")

    args = parser.parse_args()
    if args.cmd == "report":
        pairs = invoices_with_emails(args.month)
        print(render([inv for inv, _ in pairs], args.month))
    elif args.cmd == "export":
        exporter = ExporterFactory.create(args.format)
        content = exporter.export(list_invoices("all"))
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(content)
    elif args.cmd == "sync":
        sync_remote()
    elif args.cmd == "create":
        if args.strict_validation and (not args.customer or args.amount <= 0):
            raise SystemExit("счёт отклонён валидацией")
        print(f"создан счёт для {args.customer} на {args.amount}")
