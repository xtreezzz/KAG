package com.example.commerce;

import java.util.ArrayList;
import java.util.List;

public class Customer {
    private final String id;
    public Customer(String id) { this.id = id; }
    public String getId() { return id; }
}

class Product {
    private final String sku;
    private final double price;
    public Product(String sku, double price) { this.sku = sku; this.price = price; }
    public String getSku() { return sku; }
    public double getPrice() { return price; }
}

class OrderItem {
    private final Product product;
    private final int quantity;
    public OrderItem(Product product, int quantity) { this.product = product; this.quantity = quantity; }
    public Product getProduct() { return product; }
    public int getQuantity() { return quantity; }
}

class Order {
    private final String customerId;
    private final List<OrderItem> items = new ArrayList<>();
    public Order(String customerId) { this.customerId = customerId; }
    public void addItem(OrderItem item) { items.add(item); }
    public List<OrderItem> getItems() { return items; }
    public String getCustomerId() { return customerId; }
}
