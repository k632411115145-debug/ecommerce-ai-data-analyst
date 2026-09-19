# E-commerce Agent Codebook

## Scope

This database was built from the five transformed CSV files supplied for the project.

Important: these CSVs are not the original raw normalized Kaggle tables. In the supplied files, `orders`, `order_items`, `payments`, and `customers` each contain 89,316 rows and are aligned one row per order/customer. The `products` CSV contains repeated product rows, so exact duplicate product records were removed before loading into SQLite.

Use `analysis_base` for most analysis because it is intentionally one row per order in this transformed dataset.

## Table relationships

- `orders.customer_id -> customers.customer_id`
- `order_items.order_id -> orders.order_id`
- `order_items.product_id -> products.product_id`
- `payments.order_id -> orders.order_id`

## orders

- `order_id`: unique order identifier.
- `customer_id`: customer identifier.
- `order_status`: status such as delivered, shipped, canceled, processing, invoiced.
- `order_purchase_timestamp`: time the order was placed.
- `order_approved_at`: time the seller/payment side approved the order.
- `order_delivered_timestamp`: actual delivery timestamp.
- `order_estimated_delivery_date`: promised/estimated delivery date.

## order_items

- `order_id`: order identifier.
- `product_id`: product identifier.
- `seller_id`: seller identifier.
- `price`: selling price attached to the retained order-item row.
- `shipping_charges`: shipping charge attached to the retained row.

Important limitation: the original Kaggle schema supports multiple items per order via an `order_item_id`, but the supplied transformed CSV no longer contains `order_item_id` and has exactly one row per order. Do not use this transformed database to infer basket size or number of line items per order.

## payments

- `order_id`: order identifier.
- `payment_sequential`: payment sequence number.
- `payment_type`: credit_card, wallet, voucher, debit_card, etc.
- `payment_installments`: number of installments.
- `payment_value`: transaction/payment value.

Important limitation: the original schema can contain multiple payment rows per order, but the supplied transformed CSV has one payment row per order. Treat payment-level totals cautiously and do not infer the number of payments per order from this transformed table.

## customers

- `customer_id`: unique customer identifier in this transformed dataset.
- `customer_zip_code_prefix`: ZIP/postal prefix.
- `customer_city`: city.
- `customer_state`: state.

## products

- `product_id`: unique product identifier after deduplication.
- `product_category_name`: category.
- `product_weight_g`: weight in grams.
- `product_length_cm`: length in cm.
- `product_height_cm`: height in cm.
- `product_width_cm`: width in cm.

## Recommended analysis rules for the agent

1. Prefer querying `analysis_base` unless a specific normalized table is needed.
2. Use only read-only SQL (`SELECT`, `WITH`).
3. Never invent values; every numerical claim must come from executed SQL.
4. State the metric explicitly. Do not use the generic word "revenue" without defining whether it means `SUM(price)` or `SUM(payment_value)`.
5. Do not claim basket size, item count per order, or number of payment transactions per order from this transformed database.
6. When analyzing delivery performance, compare actual delivery timestamp with estimated delivery date only for rows where actual delivery exists.
7. When analyzing sales/product performance, consider filtering to delivered orders and state that choice.
8. For a chart request, return the SQL result as structured rows plus a chart specification (`bar`, `line`, `scatter`, etc.); the Streamlit layer should render the chart.
