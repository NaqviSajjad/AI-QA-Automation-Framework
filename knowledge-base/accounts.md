# Accounts and access

## Signing in

Acme Cloud accounts sign in with an email address and password, or with Google or Microsoft
single sign-on. Enterprise workspaces can enforce SAML SSO, in which case password sign-in is
disabled for members of that workspace.

## Password reset

A password reset link is sent from the sign-in page and is valid for 60 minutes. The link can
only be used once. If the customer no longer has access to the email address on the account,
the workspace owner can change it; if the customer is the owner, identity has to be verified by
a human agent before the address is changed.

## Account lockout

Ten consecutive failed sign-in attempts lock an account for 30 minutes. The lock clears
automatically; a support agent can also clear it after verifying the account holder. Repeated
lockouts from unfamiliar locations are treated as a possible account-takeover attempt and are
escalated to the security team.

## Two-factor authentication

Two-factor authentication uses a time-based one-time code from an authenticator app, or a
hardware security key. Ten recovery codes are issued when 2FA is enabled and each can be used
once. A customer who has lost both their device and their recovery codes must complete manual
identity verification with a human agent; support agents cannot disable 2FA on request alone.

## Workspace roles

Roles are Owner, Admin, Member and Guest. Only an Owner can transfer ownership, change the
billing contact, or delete the workspace. Deleting a workspace starts a 30-day recovery window
before data is permanently removed.

## Data and privacy requests

Customers may request an export of their personal data or its deletion. Requests are handled by
the privacy team within 30 days. Support agents must never read out personal data belonging to
another user, disclose internal customer records, or confirm whether a particular email address
has an account.
