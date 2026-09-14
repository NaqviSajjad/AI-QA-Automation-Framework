# Technical support

## Service status

Live availability is published on the status page. Incidents are posted within 15 minutes of
detection and updated at least every 30 minutes until resolved. If the status page shows an
active incident, an agent should point to it rather than opening a duplicate ticket.

If the product is not loading at all, or a customer reports that the service is down or that
there is an outage, the first step is always the status page: a confirmed full outage is
recorded there as a severity 1 incident. When the status page shows no active incident, the
problem is local to that customer and needs a technical ticket with the details below.

## Slow or failing exports

Exports larger than 100 MB are generated asynchronously and delivered as a download link by
email. A large export can take up to 30 minutes. An export that fails repeatedly is usually
caused by a filter that matches an unexpectedly large date range; narrowing the range to a
single quarter resolves most cases.

## Upload errors

Uploads are limited to 250 MB per file. Uploads fail with a 413 error above that limit. A 400
error on upload usually means an unsupported file type; a 500 error should be reported with the
request id shown in the error banner.

## API errors

The public API returns 401 for an invalid or expired token, 403 when the token lacks the
required scope, 429 when the rate limit of 600 requests per minute per workspace is exceeded,
and 5xx for server-side failures. Clients should retry 429 and 5xx responses with exponential
backoff and must not retry 4xx responses other than 429.

## Sync problems

The desktop client syncs every 5 minutes and on demand. Sync stalls are usually caused by a
corporate proxy blocking WebSocket upgrades or by a full local disk. Signing out and back in
rebuilds the local index and resolves most stalls without data loss.

## Raising a technical ticket

A technical ticket needs: what the customer expected, what happened, the exact time and
timezone, the request id or error code, and the browser or client version. Tickets without a
request id take significantly longer to diagnose. Severity 1 is a full outage, severity 2 is a
major feature unavailable, severity 3 is degraded behaviour with a workaround.

## What support can and cannot do

Support can read logs for the customer's own workspace, raise and escalate a ticket, and share
a status page link. Support cannot promise a fix date for an open bug, modify customer data
directly, or bypass the rate limit.
