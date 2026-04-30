// --- State ---
let defaultConfig = {
  agent_available: true,
  agent_error: '',
  default_model: 'gpt-4o',
  default_base_url: '',
  default_max_iterations: 30,
};
let uploadedFiles = [];
let isProcessing = false;
let currentEventSource = null;

// --- Server Config ---
async function loadServerConfig() {
  try {
    const resp = await fetch('/api/config');
    if (!resp.ok) return;
    defaultConfig = await resp.json();
    const modelInput = document.getElementById('settModel');
    const baseUrlInput = document.getElementById('settBaseUrl');
    const maxIterInput = document.getElementById('settMaxIter');
    if (modelInput) modelInput.placeholder = defaultConfig.default_model || 'gpt-4o';
    if (baseUrlInput) baseUrlInput.placeholder = defaultConfig.default_base_url || 'https://api.openai.com/v1';
    if (maxIterInput) maxIterInput.value = localStorage.getItem('vlm_max_iter') || defaultConfig.default_max_iterations || 30;
    updateCurrentModel();
    if (!defaultConfig.agent_available) {
      const container = createAssistantMessage();
      handleEvent({
        type: 'error',
        data: {message: 'Agent not available: ' + (defaultConfig.agent_error || 'unknown import error')}
      }, container);
    }
  } catch (e) {
    console.warn('Failed to load server config', e);
  }
}

// --- Settings ---
function loadSettings() {
  return {
    apiKey: localStorage.getItem('vlm_api_key') || '',
    baseUrl: localStorage.getItem('vlm_base_url') || defaultConfig.default_base_url || '',
    model: localStorage.getItem('vlm_model') || defaultConfig.default_model || 'gpt-4o',
    maxIter: parseInt(localStorage.getItem('vlm_max_iter') || defaultConfig.default_max_iterations || 30),
    reasoning: localStorage.getItem('vlm_reasoning') !== 'false',
  };
}

function updateCurrentModel() {
  const modelEl = document.getElementById('currentModel');
  if (!modelEl) return;

  const model = loadSettings().model;
  modelEl.textContent = model;
  modelEl.title = model;
}

function saveSettings() {
  localStorage.setItem('vlm_api_key', document.getElementById('settApiKey').value);
  localStorage.setItem('vlm_base_url', document.getElementById('settBaseUrl').value);
  localStorage.setItem('vlm_model', document.getElementById('settModel').value);
  localStorage.setItem('vlm_max_iter', document.getElementById('settMaxIter').value);
  localStorage.setItem('vlm_reasoning', document.getElementById('settReasoning').classList.contains('on'));
  updateCurrentModel();
}
function openSettings() {
  const s = loadSettings();
  document.getElementById('settApiKey').value = s.apiKey;
  document.getElementById('settBaseUrl').value = s.baseUrl;
  document.getElementById('settModel').value = s.model;
  document.getElementById('settMaxIter').value = s.maxIter;
  const tog = document.getElementById('settReasoning');
  tog.classList.toggle('on', s.reasoning);
  document.getElementById('settingsOverlay').classList.add('open');
}
function closeSettings() {
  saveSettings();
  document.getElementById('settingsOverlay').classList.remove('open');
}

// --- UI Helpers ---
function scrollToBottom() {
  const c = document.getElementById('chatContainer');
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeLightbox(); closeSettings(); } });

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function clearChat() {
  if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
  document.getElementById('chatMessages').innerHTML = `
    <div class="welcome" id="welcome">
      <h2>SWE-Vision Agent</h2>
      <p>Upload an image and ask a question. The agent will use a Jupyter notebook to analyze it step-by-step.</p>
    </div>`;
  isProcessing = false;
  document.getElementById('sendBtn').disabled = false;
}

// --- File Upload ---
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) addFile(f);
  fileInput.value = '';
});

function addFile(file) {
  if (!file.type.startsWith('image/')) return;
  uploadedFiles.push(file);
  renderPreviews();
}
function removeFile(idx) {
  uploadedFiles.splice(idx, 1);
  renderPreviews();
}
function renderPreviews() {
  const el = document.getElementById('uploadPreview');
  el.innerHTML = '';
  el.classList.toggle('has-files', uploadedFiles.length > 0);
  uploadedFiles.forEach((f, i) => {
    const thumb = document.createElement('div');
    thumb.className = 'upload-thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    const btn = document.createElement('button');
    btn.className = 'remove-btn';
    btn.textContent = '\u00d7';
    btn.onclick = () => removeFile(i);
    thumb.append(img, btn);
    el.append(thumb);
  });
}

// Drag & Drop
document.addEventListener('dragover', e => { e.preventDefault(); document.getElementById('dragOverlay').style.display = 'flex'; });
document.addEventListener('dragleave', e => { if (e.relatedTarget === null) document.getElementById('dragOverlay').style.display = 'none'; });
document.addEventListener('drop', e => {
  e.preventDefault();
  document.getElementById('dragOverlay').style.display = 'none';
  for (const f of e.dataTransfer.files) addFile(f);
});

// --- Markdown ---
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, {language: lang}).value;
    return hljs.highlightAuto(code).value;
  }
});

function renderMarkdown(text) {
  try { return marked.parse(text); } catch { return escapeHtml(text); }
}
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// --- Message Rendering ---
function addUserMessage(text, files) {
  const wel = document.getElementById('welcome');
  if (wel) wel.remove();

  const msg = document.createElement('div');
  msg.className = 'message message-user';
  let html = `<div class="bubble">${escapeHtml(text)}</div>`;
  if (files && files.length > 0) {
    html += '<div class="msg-images">';
    for (const f of files) {
      html += `<img src="${URL.createObjectURL(f)}" onclick="openLightbox(this.src)">`;
    }
    html += '</div>';
  }
  msg.innerHTML = html;
  document.getElementById('chatMessages').append(msg);
  scrollToBottom();
}

function createAssistantMessage() {
  const msg = document.createElement('div');
  msg.className = 'message message-assistant';
  msg.innerHTML = `
    <div class="avatar avatar-assistant">V</div>
    <div class="assistant-content" id="assistantContent_${Date.now()}"></div>
  `;
  document.getElementById('chatMessages').append(msg);
  scrollToBottom();
  return msg.querySelector('.assistant-content');
}

function addLoadingIndicator(container) {
  const el = document.createElement('div');
  el.className = 'loading-indicator';
  el.id = 'loadingIndicator';
  el.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div><span>Agent is thinking...</span>';
  container.append(el);
  scrollToBottom();
}
function removeLoadingIndicator() {
  const el = document.getElementById('loadingIndicator');
  if (el) el.remove();
}
function updateLoadingText(text) {
  const el = document.getElementById('loadingIndicator');
  if (el) el.querySelector('span').textContent = text;
}

// --- Event Handlers ---
function handleEvent(event, container) {
  switch (event.type) {
    case 'status':
      updateLoadingText(event.data.message);
      break;

    case 'thinking': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'step-card thinking-card';
      const content = event.data.content || '';
      const preview = content.substring(0, 80) + (content.length > 80 ? '...' : '');
      card.innerHTML = `
        <div class="thinking-header" onclick="this.parentElement.classList.toggle('open')">
          <span class="arrow">\u25b8</span>
          <span>Thinking</span>
          <span style="color:var(--text-muted);font-weight:400;font-size:12px;margin-left:auto;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(preview)}</span>
        </div>
        <div class="thinking-body">${escapeHtml(content)}</div>
      `;
      container.append(card);
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'assistant_text': {
      removeLoadingIndicator();
      const div = document.createElement('div');
      div.className = 'assistant-text';
      div.innerHTML = renderMarkdown(event.data.content);
      container.append(div);
      container.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'tool_call': {
      removeLoadingIndicator();
      if (event.data.name === 'finish') {
        break;
      }
      const card = document.createElement('div');
      card.className = 'step-card toolcall-card';
      const code = event.data.code || event.data.arguments || '';
      const highlighted = code ? hljs.highlight(code, {language: 'python'}).value : '';
      card.innerHTML = `
        <div class="toolcall-label">
          <span class="fn-icon">\u26a1</span>
          <span class="fn-badge">${escapeHtml(event.data.name)}</span>
        </div>
        <div class="toolcall-code">
          <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          <pre><code>${highlighted}</code></pre>
        </div>
      `;
      container.append(card);
      updateLoadingText('Running code...');
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'tool_result': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'step-card result-card' + (event.data.is_error ? ' error' : '');
      const label = event.data.is_error ? 'Error' : 'Output';
      let html = `
        <div class="result-label">
          <span class="res-badge">${label}</span>
        </div>
      `;
      if (event.data.output) {
        html += `<div class="result-output">${escapeHtml(event.data.output)}</div>`;
      }
      if (event.data.images && event.data.images.length > 0) {
        html += '<div class="result-images">';
        for (const src of event.data.images) {
          html += `<img src="${src}" onclick="openLightbox(this.src)" loading="lazy">`;
        }
        html += '</div>';
      }
      card.innerHTML = html;
      container.append(card);
      addLoadingIndicator(container);
      scrollToBottom();
      break;
    }

    case 'finish': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'finish-card';
      card.innerHTML = `
        <div class="finish-label">\u2713 Final Answer</div>
        <div class="finish-text">${renderMarkdown(event.data.answer)}</div>
      `;
      container.append(card);
      container.querySelectorAll('.finish-text pre code').forEach(el => hljs.highlightElement(el));
      scrollToBottom();
      break;
    }

    case 'error': {
      removeLoadingIndicator();
      const card = document.createElement('div');
      card.className = 'error-card';
      card.textContent = event.data.message;
      container.append(card);
      scrollToBottom();
      break;
    }

    case 'done':
      removeLoadingIndicator();
      break;
  }
}

function copyCode(btn) {
  const code = btn.parentElement.querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  });
}

// --- Send Message ---
async function sendMessage() {
  const input = document.getElementById('promptInput');
  const prompt = input.value.trim();
  if (!prompt || isProcessing) return;

  isProcessing = true;
  document.getElementById('sendBtn').disabled = true;

  const settings = loadSettings();
  const filesToSend = [...uploadedFiles];

  addUserMessage(prompt, filesToSend);

  input.value = '';
  input.style.height = 'auto';
  uploadedFiles = [];
  renderPreviews();

  const container = createAssistantMessage();
  addLoadingIndicator(container);

  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('model', settings.model);
  formData.append('api_key', settings.apiKey);
  formData.append('base_url', settings.baseUrl);
  formData.append('reasoning', settings.reasoning);
  formData.append('max_iterations', settings.maxIter);
  for (const f of filesToSend) formData.append('images', f);

  try {
    const resp = await fetch('/api/chat', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      handleEvent({type: 'error', data: {message: err.error || 'Request failed'}}, container);
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      return;
    }
    const { session_id } = await resp.json();

    const es = new EventSource(`/api/stream/${session_id}`);
    currentEventSource = es;

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);
      handleEvent(event, container);
      if (event.type === 'finish' || event.type === 'error' || event.type === 'done') {
        es.close();
        currentEventSource = null;
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
      }
    };
    es.onerror = () => {
      es.close();
      currentEventSource = null;
      if (isProcessing) {
        handleEvent({type: 'error', data: {message: 'Connection lost. The agent may still be running in the background.'}}, container);
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
      }
    };
  } catch (e) {
    handleEvent({type: 'error', data: {message: 'Network error: ' + e.message}}, container);
    isProcessing = false;
    document.getElementById('sendBtn').disabled = false;
  }
}

// --- Init ---
async function initApp() {
  await loadServerConfig();
  updateCurrentModel();
  document.getElementById('promptInput').focus();
}
initApp();
