"""Churn analysis helpers for the commerce domain."""

from datetime import datetime, timedelta
from typing import Iterable, List, Dict


class Customer:
    """Customer entity with a stable customer_id."""

    def __init__(self, customer_id: str, created_at: datetime) -> None:
        self.customer_id = customer_id
        self.created_at = created_at


class Order:
    """Order transaction associated with a Customer."""

    def __init__(self, order_id: str, customer_id: str, created_at: datetime) -> None:
        self.order_id = order_id
        self.customer_id = customer_id
        self.created_at = created_at


def is_churned(last_order_at: datetime, as_of: datetime, days: int = 90) -> bool:
    """Return True if a customer has no orders in the last `days` days."""
    return (as_of - last_order_at) > timedelta(days=days)


def compute_churn(customers: Iterable[Customer], orders: Iterable[Order], as_of: datetime) -> Dict[str, bool]:
    """Compute churn status per customer_id."""
    last_order_by_customer: Dict[str, datetime] = {}
    for order in orders:
        if order.customer_id not in last_order_by_customer:
            last_order_by_customer[order.customer_id] = order.created_at
        else:
            last_order_by_customer[order.customer_id] = max(last_order_by_customer[order.customer_id], order.created_at)

    result: Dict[str, bool] = {}
    for customer in customers:
        last_order_at = last_order_by_customer.get(customer.customer_id, customer.created_at)
        result[customer.customer_id] = is_churned(last_order_at, as_of)
    return result


def retention_rate(churn_flags: Dict[str, bool]) -> float:
    """Retention is the inverse of churn."""
    total = len(churn_flags)
    if total == 0:
        return 0.0
    retained = sum(1 for v in churn_flags.values() if not v)
    return retained / total
