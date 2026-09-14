# Payments

## Accepted payment methods

Acme Cloud accepts Visa, Mastercard and American Express, SEPA direct debit for customers with
a billing address in the SEPA area, and PayPal on monthly plans. Enterprise customers may pay
annually by bank transfer against a purchase order.

## Why a card payment is declined

The most common reasons a payment fails are: insufficient funds, an expired card, an incorrect
CVC or postcode, a card that does not support recurring international payments, a bank fraud
block on a first-time merchant, or a 3-D Secure challenge that was never completed.

Acme Cloud never receives the specific decline reason from the issuing bank beyond a generic
code, so the customer's bank is the only party that can confirm why a card was refused.

## 3-D Secure and strong customer authentication

Payments from EEA and UK cards may require Strong Customer Authentication. When the bank
requests it, the customer sees a verification step from their bank. If that step is abandoned
the payment stays in a pending state for up to 30 minutes and then fails. Pending
authorisations may appear on a bank statement as a temporary hold and drop off within 7 days
without ever being captured.

## Retry schedule

A failed recurring charge is retried automatically on days 1, 3 and 7 after the due date.
A customer can also trigger an immediate retry from Settings, Billing, Retry payment, once a
working payment method is saved.

## Updating a payment method

Payment methods are updated in Settings, Billing, Payment methods. The change takes effect for
the next charge attempt immediately. Card details are stored by the payment provider and are
never visible to Acme Cloud staff; support agents can only see the card brand, the last four
digits and the expiry month.

## What support can and cannot do

Support can check whether a payment attempt succeeded, failed or is pending, see the failure
code, and trigger a retry. Support cannot charge a card manually, view or enter full card
numbers, override a bank decline, or remove a 3-D Secure requirement.
