# Glossary

**Customer** is a person or organization that buys Products.
**Order** is a purchase request made by a Customer.
**OrderItem** is a line inside an Order and references a Product.
**Product** is an item offered for sale.

Customer is_a Entity.
Order is_a Transaction.
OrderItem is_a LineItem.
Product is_a CatalogItem.

Customer part_of Market.
Order part_of Revenue.
