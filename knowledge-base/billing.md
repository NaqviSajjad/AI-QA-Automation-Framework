# Billing

Acme Cloud bills all subscriptions in arrears on the first calendar day of each month.
Invoices are generated at 02:00 UTC and emailed to the billing contact on the account.

## Duplicate charges

A duplicate charge occurs when the same invoice is captured twice against the same payment
method, usually because a card retry succeeded after the original authorisation had already
settled. Duplicate charges are detected automatically by the billing reconciliation job that
runs every 24 hours.

When a duplicate charge is confirmed, the extra amount is reversed to the original payment
method. Card reversals reach the customer's statement within 5 to 10 business days depending
on the issuing bank. SEPA direct debit reversals take 3 to 5 business days.

A support agent should always confirm the invoice number and the charge date before promising
a reversal. A charge that appears twice on a bank statement but only once on the invoice list
is usually a pending authorisation that will drop off within 7 days, not a duplicate charge.

## Invoice questions

Every invoice shows the billing period, the plan, the seat count, applicable VAT and the
payment method used. Customers on the Business and Enterprise plans can download invoices as
PDF from Settings, Billing, Invoice history. Invoice history is retained for 7 years.

VAT is charged according to the billing address on the account. A customer who supplies a
valid EU VAT identification number is billed under the reverse-charge mechanism and the VAT
line becomes zero from the next billing period onwards. VAT already charged on past invoices
is not retroactively refunded.

## Proration

Changing plan mid-cycle produces a prorated line on the next invoice: unused time on the old
plan is credited and the new plan is charged from the day of the change. Seat additions are
prorated to the day. Seat removals take effect at the end of the billing period and are not
credited mid-cycle.

## Failed billing and dunning

If a charge fails, the system retries on days 1, 3 and 7. The billing contact receives an
email after each failed attempt. If all three retries fail, the account moves to a grace
period of 14 days during which the service stays fully available. After the grace period the
account is downgraded to read-only until payment succeeds. Data is never deleted during
downgrade.

## What support can and cannot do

Support can look up billing history, resend an invoice, correct a billing address, and start
a duplicate-charge investigation. Support cannot change a settled invoice, waive VAT, or
disclose full card numbers. Any request touching card data must be handled through the
payment provider's secure portal.
