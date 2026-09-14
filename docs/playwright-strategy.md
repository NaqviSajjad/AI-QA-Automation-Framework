# Playwright strategy

Playwright is the primary automation framework here. Its job is the user journey and the
capture of what the assistant actually said. It does not score anything.

## Selectors

`data-testid` only. Never CSS classes, never text content.

```python
chatbot_page.send_message("I was charged twice this month.")
response = chatbot_page.get_latest_response()
```

The application exposes: `chat-input`, `send-button`, `ai-response`, `user-message`,
`loading-indicator`, `error-message`, `retry-button`, `conversation-id`,
`escalation-indicator`, `category-badge`, `conversation-history`, `reset-button`,
`logout-button`, plus the login form's `email-input`, `password-input`, `login-button`.

Text-based selectors are a particularly bad idea in an AI product: the copy on screen is
generated, so a locator bound to it is bound to model output.

## Page objects

```
pages/
  base_page.py            navigation, testid lookup, waits, screenshots
  login_page.py
  chatbot_page.py         send_message, get_latest_response, is_escalated, ...
  conversation_page.py    transcript-level assertions
  components/
    chat_input.py
    chat_message.py       parses the category/priority metadata line
    conversation_history.py
```

Two rules:

1. **Tests never see a selector.** They call intention-revealing methods.
2. **Page objects never assert.** They expose state; the test decides what is correct.

A test reads as business intent:

```python
def test_billing_complaint(chatbot_page):
    chatbot_page.send_message("I was charged twice this month.")
    assert chatbot_page.get_latest_response()
    assert chatbot_page.get_category() == "billing"
```

## Fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `app_server` | session | Boots the app on a free port, waits for `/api/health`, tears it down |
| `authenticated_state` | session | Signs in once, stores the cookie |
| `authenticated_context` | function | Fresh browser context from that stored state |
| `chatbot_page` | function | Signed-in chat page, open and ready |
| `login_page` | function | Clean, signed-out login page |

`pytest` is the only command anyone needs: the suite starts its own application. Point it
at a running instance with `APP_AUTO_START=false`.

Signing in through the UI in every test would re-test the login form fifty times and add
a second of latency to each case. `tests/ui/test_login.py` owns that behaviour; every
other test starts already authenticated.

One consequence worth knowing about: the logout test signs in through the UI *on purpose*,
because logging out invalidates the server-side session that the shared state depends on.
That is exactly the hidden coupling that produces "passes alone, fails in the suite"
flakiness, and it is called out in a comment where it lives.

## Waiting, and why these tests are not flaky

The response wait is a statement about the outcome, not about an animation:

```python
self.page.wait_for_function(
    """(target) => {
        const answers = document.querySelectorAll('[data-testid="ai-response"]').length;
        const error = document.querySelector('[data-testid="error-message"]');
        return answers >= target || (error && !error.classList.contains('hidden'));
    }""",
    arg=expected_count,
)
```

It waits for *a new answer or a visible error*. It never sleeps, never polls a spinner,
and cannot hang forever on a failure path.

Other flake-control choices:

* no `time.sleep` anywhere in the suite;
* every assertion is on state the user can see, not on intermediate DOM;
* AI answers are asserted for *properties* (non-empty, minimum length, category), never
  for exact strings — exact-match on generated text is the single largest source of
  flaky AI tests;
* semantic judgement is delegated to the evaluation layer, which has thresholds and
  reasons rather than equality.

## Error paths

An AI feature fails in ways a CRUD screen does not, so each mode is triggered
deliberately rather than waited for:

| Failure | How it is triggered |
|---|---|
| Model timeout | `__FORCE_TIMEOUT__` in the message → provider raises → API 504 |
| Empty completion | `__FORCE_EMPTY__` → API 502 |
| Upstream failure | `__FORCE_ERROR__` → API 502 |
| Network failure | `page.route(...).abort()` |
| Transient failure then success | a route handler that fails once, then continues |

The last one asserts that the retry button actually recovers, and that the request was
made exactly twice.

## Cross-browser

```bash
pytest --browser chromium --browser firefox --browser webkit
```

Runs in the nightly matrix. Nothing in the page objects is browser-specific; the only
browser-sensitive assertion is the native form-validation message, which is read through
`el.validationMessage` rather than by matching English text.

## Artefacts

Configured through pytest-playwright, written under `reports/`:

```bash
pytest --tracing retain-on-failure --video retain-on-failure --screenshot only-on-failure
```

* HTML report — `reports/pytest-report.html` (always)
* JUnit XML — `--junitxml=reports/junit.xml`, consumed by the report generator
* Trace, video, screenshot — on failure
* Structured JSON logs carrying the conversation id and trace id, so a UI failure can be
  correlated with the agent trace that produced it

## Markers

```
smoke ui api llm rag agent prompt regression security performance flagship slow
```

`-m smoke` is the pull-request critical path. `-m "not slow"` excludes the full-corpus
sweeps.
