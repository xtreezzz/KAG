-- PostgreSQL schema (inspired by common sample DBs like Northwind)

CREATE TABLE customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    email TEXT,
    country TEXT
);

CREATE TABLE products (
    product_id VARCHAR(32) PRIMARY KEY,
    sku TEXT NOT NULL,
    name TEXT NOT NULL,
    price NUMERIC(10,2) NOT NULL
);

CREATE TABLE orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    status TEXT,
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_item_id VARCHAR(32) PRIMARY KEY,
    order_id VARCHAR(32) NOT NULL,
    product_id VARCHAR(32) NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    CONSTRAINT fk_items_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT fk_items_product FOREIGN KEY (product_id) REFERENCES products(product_id)
);
