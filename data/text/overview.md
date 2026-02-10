# Context Mining for LLM

This project builds a **Knowledge-Augmented Graph (KAG)** from text, code, and SQL.
A KAG helps a model resolve **entities**, **definitions**, and **relationships** across sources.

## Domain snapshot
We analyze a small commerce domain with **Customer**, **Order**, **OrderItem**, and **Product**.
Churn is the event when a Customer stops placing Orders for 90 days.
Retention is the inverse of churn.

## Notes (RU)
KAG помогает связать термины из документов и кода.
Сущность "Покупатель" соответствует Customer.
