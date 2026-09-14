"""HTML for the application under test.

Selectors are `data-testid` attributes only. Page objects never bind to CSS
classes or text, so restyling the app cannot break the suite.
"""

from __future__ import annotations

_STYLE = """
:root { color-scheme: light dark; --bg:#f6f7f9; --fg:#12151a; --muted:#5b6472;
        --card:#ffffff; --line:#e2e6ec; --accent:#2f6feb; --warn:#b4530a; --err:#b3261e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1216; --fg:#e8eaee; --muted:#98a2b3; --card:#171b21; --line:#262c35;
          --accent:#6c9bff; --warn:#e0913f; --err:#ff6b6b; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,
       BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width: 760px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 20px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.row { display: flex; gap: 8px; align-items: center; }
label { display:block; font-size:13px; color:var(--muted); margin: 10px 0 4px; }
input, textarea { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px;
       background:var(--bg); color:var(--fg); font: inherit; }
button { padding:10px 16px; border:0; border-radius:8px; background:var(--accent); color:#fff;
       font: inherit; font-weight:600; cursor:pointer; }
button.secondary { background:transparent; color:var(--muted); border:1px solid var(--line); }
button:disabled { opacity:.55; cursor:not-allowed; }
.history { display:flex; flex-direction:column; gap:12px; min-height:220px; max-height:52vh;
       overflow-y:auto; padding-right:4px; margin-bottom:14px; }
.msg { padding:10px 12px; border-radius:10px; max-width:88%; white-space:pre-wrap; }
.msg.user { align-self:flex-end; background:var(--accent); color:#fff; }
.msg.ai { align-self:flex-start; background:var(--bg); border:1px solid var(--line); }
.meta { font-size:12px; color:var(--muted); margin-top:6px; display:flex; gap:10px; flex-wrap:wrap; }
.bar { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:12px;
       font-size:12px; color:var(--muted); }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
.badge.escalated { color:var(--warn); border-color:var(--warn); }
.error { color:var(--err); border:1px solid var(--err); border-radius:8px; padding:10px 12px;
       margin-bottom:12px; font-size:14px; }
.hidden { display:none !important; }
.dots::after { content:"..."; animation: dots 1s steps(4,end) infinite; }
@keyframes dots { 0%{content:"."} 33%{content:".."} 66%{content:"..."} }
"""

LOGIN_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in - Acme Cloud Support</title><style>{_STYLE}</style></head>
<body><div class="wrap">
  <h1>Acme Cloud</h1>
  <p class="sub">Sign in to contact customer support.</p>
  <div class="card">
    <form id="login-form" data-testid="login-form">
      <label for="email">Email</label>
      <input id="email" name="email" type="email" autocomplete="username" data-testid="email-input" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" data-testid="password-input" required>
      <div style="height:14px"></div>
      <button type="submit" data-testid="login-button">Sign in</button>
    </form>
    <div id="login-error" class="error hidden" data-testid="error-message" role="alert"></div>
  </div>
</div>
<script>
const form = document.getElementById('login-form');
const errorBox = document.getElementById('login-error');
form.addEventListener('submit', async (event) => {{
  event.preventDefault();
  errorBox.classList.add('hidden');
  const button = form.querySelector('[data-testid="login-button"]');
  button.disabled = true;
  try {{
    const response = await fetch('/api/auth/login', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        email: document.getElementById('email').value,
        password: document.getElementById('password').value
      }})
    }});
    if (!response.ok) {{
      const body = await response.json().catch(() => ({{}}));
      errorBox.textContent = body.error || 'Invalid email or password';
      errorBox.classList.remove('hidden');
      return;
    }}
    window.location.href = '/chat';
  }} catch (err) {{
    errorBox.textContent = 'Network error. Please try again.';
    errorBox.classList.remove('hidden');
  }} finally {{
    button.disabled = false;
  }}
}});
</script>
</body></html>
"""

CHAT_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Support AI - Acme Cloud</title><style>{_STYLE}</style></head>
<body><div class="wrap">
  <h1>Customer Support AI</h1>
  <p class="sub">Describe your issue and the assistant will help or pass you to a specialist.</p>

  <div class="bar">
    <span>Conversation: <span data-testid="conversation-id" id="conversation-id">-</span></span>
    <span class="row">
      <span class="badge hidden" data-testid="category-badge" id="category-badge"></span>
      <span class="badge escalated hidden" data-testid="escalation-indicator" id="escalation-indicator">Escalated to a human</span>
    </span>
  </div>

  <div class="card">
    <div id="error-box" class="error hidden" data-testid="error-message" role="alert"></div>
    <button id="retry-button" class="secondary hidden" data-testid="retry-button">Retry</button>

    <div class="history" id="history" data-testid="conversation-history"></div>

    <div id="loading" class="sub hidden" data-testid="loading-indicator">
      <span class="dots">Assistant is typing</span>
    </div>

    <form id="chat-form">
      <textarea id="chat-input" data-testid="chat-input" rows="3"
                placeholder="Describe your issue..." aria-label="Describe your issue"></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <button type="submit" data-testid="send-button" id="send-button">Send</button>
        <button type="button" class="secondary" data-testid="reset-button" id="reset-button">New conversation</button>
        <button type="button" class="secondary" data-testid="logout-button" id="logout-button">Sign out</button>
      </div>
    </form>
  </div>
</div>
<script>
let conversationId = null;
let lastMessage = null;

const history = document.getElementById('history');
const input = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const loading = document.getElementById('loading');
const errorBox = document.getElementById('error-box');
const retryButton = document.getElementById('retry-button');
const escalation = document.getElementById('escalation-indicator');
const categoryBadge = document.getElementById('category-badge');

function addMessage(role, text, meta) {{
  const node = document.createElement('div');
  node.className = 'msg ' + (role === 'user' ? 'user' : 'ai');
  node.setAttribute('data-testid', role === 'user' ? 'user-message' : 'ai-response');
  node.textContent = text;
  if (meta) {{
    const metaNode = document.createElement('div');
    metaNode.className = 'meta';
    metaNode.setAttribute('data-testid', 'ai-response-meta');
    metaNode.textContent = meta;
    node.appendChild(metaNode);
  }}
  history.appendChild(node);
  history.scrollTop = history.scrollHeight;
}}

function showError(message) {{
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
  retryButton.classList.remove('hidden');
}}

function clearError() {{
  errorBox.classList.add('hidden');
  retryButton.classList.add('hidden');
}}

async function send(message) {{
  clearError();
  lastMessage = message;
  loading.classList.remove('hidden');
  sendButton.disabled = true;
  try {{
    const response = await fetch('/api/chat', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ message: message, conversation_id: conversationId }})
    }});
    const body = await response.json().catch(() => ({{}}));
    if (!response.ok) {{
      showError(body.message || body.error || ('Request failed with status ' + response.status));
      return;
    }}
    conversationId = body.conversation_id;
    document.getElementById('conversation-id').textContent = conversationId;
    if (body.category) {{
      categoryBadge.textContent = body.category;
      categoryBadge.classList.remove('hidden');
    }}
    escalation.classList.toggle('hidden', !body.requires_escalation);
    addMessage('assistant', body.response, 'category: ' + body.category + ' | priority: ' + body.priority);
  }} catch (err) {{
    showError('Network error. The assistant could not be reached.');
  }} finally {{
    loading.classList.add('hidden');
    sendButton.disabled = false;
  }}
}}

document.getElementById('chat-form').addEventListener('submit', (event) => {{
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {{ showError('Please describe your issue before sending.'); return; }}
  addMessage('user', message);
  input.value = '';
  send(message);
}});

retryButton.addEventListener('click', () => {{ if (lastMessage) send(lastMessage); }});

document.getElementById('reset-button').addEventListener('click', async () => {{
  if (conversationId) {{
    await fetch('/api/conversations/' + conversationId, {{ method: 'DELETE' }});
  }}
  conversationId = null;
  history.innerHTML = '';
  clearError();
  escalation.classList.add('hidden');
  categoryBadge.classList.add('hidden');
  document.getElementById('conversation-id').textContent = '-';
}});

document.getElementById('logout-button').addEventListener('click', async () => {{
  await fetch('/api/auth/logout', {{ method: 'POST' }});
  window.location.href = '/login';
}});
</script>
</body></html>
"""
