const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#audio-file');
const button = document.querySelector('#analyze');
const statusLine = document.querySelector('#status');
const player = document.querySelector('#player');
let ready = false;
let audioURL;

fetch('/api/status').then(response => response.json()).then(data => {
  ready = data.ready;
  button.disabled = !ready || !fileInput.files.length;
  statusLine.textContent = data.message;
}).catch(() => { statusLine.textContent = 'Cannot reach the model. Refresh to try again.'; });

fileInput.addEventListener('change', () => {
  if (audioURL) URL.revokeObjectURL(audioURL);
  const file = fileInput.files[0];
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty').hidden = false;
  player.hidden = !file;
  if (file) { audioURL = URL.createObjectURL(file); player.src = audioURL; }
  else { player.removeAttribute('src'); player.load(); }
  button.disabled = !ready || !file;
  if (ready) statusLine.textContent = file ? 'Ready to analyze your clip.' : 'Choose a clip to begin.';
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file || !ready) return;
  if (file.size > 40 * 1024 * 1024) { statusLine.textContent = 'Choose a file smaller than 40 MB.'; return; }
  button.disabled = true;
  fileInput.disabled = true;
  button.textContent = 'Analyzing…';
  statusLine.textContent = 'Listening to your clip…';
  const body = new FormData(); body.append('file', file);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not analyze this clip. Please try again.');
    document.querySelector('#genre').textContent = data.genre;
    const scores = document.querySelector('#scores'); scores.replaceChildren();
    data.scores.slice(0, 3).forEach(item => {
      const row = document.createElement('div'); row.className = 'score';
      const label = document.createElement('div'); label.className = 'score-label';
      const name = document.createElement('span'); name.textContent = item.genre;
      const value = document.createElement('span'); value.textContent = `${(item.score * 100).toFixed(1)}%`;
      label.append(name, value);
      const bar = document.createElement('div'); bar.className = 'bar'; bar.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${Math.max(0, Math.min(100, item.score * 100))}%`;
      bar.append(fill); row.append(label, bar); scores.append(row);
    });
    document.querySelector('#duration').textContent = `${data.seconds_analyzed} seconds analyzed across ${data.segments} excerpts.`;
    document.querySelector('#empty').hidden = true;
    document.querySelector('#result').hidden = false;
    statusLine.textContent = 'Analysis complete.';
  } catch (error) {
    document.querySelector('#result').hidden = true;
    document.querySelector('#empty').hidden = false;
    statusLine.textContent = error.message;
  } finally {
    button.textContent = 'Analyze clip'; button.disabled = !ready; fileInput.disabled = false;
  }
});
