package com.example.commerce;

import java.util.List;

public class OrderService {
    public Order createOrder(Customer customer, List<OrderItem> items) {
        Order order = new Order(customer.getId());
        for (OrderItem item : items) {
            order.addItem(item);
        }
        return order;
    }

    public double calculateTotal(Order order) {
        double total = 0.0;
        for (OrderItem item : order.getItems()) {
            total += item.getQuantity() * item.getProduct().getPrice();
        }
        return total;
    }
}
