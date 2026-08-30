const fileInput = document.querySelector('#file');
const preview = document.querySelector('#preview');
const analyzeButton = document.querySelector('#analyze');
const statusLine = document.querySelector('#status');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  analyzeButton.disabled = !file;
  if (file) { preview.src = URL.createObjectURL(file); preview.hidden = false; }
});

analyzeButton.addEventListener('click', async () => {
  const form = new FormData(); form.append('file', fileInput.files[0]);
  analyzeButton.disabled = true; statusLine.className = ''; statusLine.textContent = 'Analyzing…';
  try {
    const response = await fetch('/api/analyze', { method:'POST', body:form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Analysis failed');
    renderResult(body); statusLine.textContent = 'Analysis complete.'; loadHistory();
  } catch (error) { statusLine.className = 'error'; statusLine.textContent = error.message; }
  finally { analyzeButton.disabled = false; }
});

function renderResult(data) {
  document.querySelector('#result').hidden = false;
  document.querySelector('#label').textContent = data.quality_label.replaceAll('_', ' ');
  document.querySelector('#score').textContent = Math.round(data.quality_score);
  document.querySelector('#quality-summary').innerHTML = data.quality_summary.map(value => `<p>${escapeHtml(value)}</p>`).join('');
  document.querySelector('#issues').innerHTML = data.issues.map(issue => {
    const percent = Math.round(issue.confidence * 100);
    return `<article class="issue issue-${issue.type} ${issue.detected ? 'detected' : ''}"><div class="issue-title"><strong>${title(issue.type)}</strong><span>${percent}%</span></div><div class="confidence-track" aria-label="${escapeHtml(title(issue.type))} confidence ${percent}%"><span style="width:${percent}%"></span></div><p>${issue.detected ? `${issue.severity} severity · detected` : 'below detection threshold'}</p></article>`;
  }).join('');
  document.querySelector('#explanations').innerHTML = data.explanations.map(value => `<li>${escapeHtml(value)}</li>`).join('');
  document.querySelector('#statistics').innerHTML = Object.entries(data.statistics).map(([key,value]) => `<div class="stat"><span>${title(key)}</span><strong>${Number(value).toFixed(3)}</strong></div>`).join('');
}

async function loadHistory() {
  const container = document.querySelector('#history');
  try {
    const response = await fetch('/api/history'); if (!response.ok) throw new Error();
    const rows = await response.json();
    container.innerHTML = rows.length ? rows.map(row => `<div class="history-item"><strong>${escapeHtml(row.filename)}</strong><span>${Math.round(row.quality_score)} · ${row.quality_label.replaceAll('_',' ')}</span><time>${new Date(row.created_at).toLocaleString()}</time></div>`).join('') : '<p class="muted">No analyses yet.</p>';
  } catch { container.innerHTML = '<p class="error">Could not load history.</p>'; }
}
function title(value){ return value.replaceAll('_',' ').replace(/\b\w/g, char => char.toUpperCase()); }
function escapeHtml(value){ const node=document.createElement('span'); node.textContent=value; return node.innerHTML; }
document.querySelector('#refresh').addEventListener('click', loadHistory); loadHistory();
