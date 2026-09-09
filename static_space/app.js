const CLASSES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'];
const SAMPLE_RATE = 22050, SEGMENT_SAMPLES = 66150, FFT_SIZE = 1024, HOP = 512;
const MEL_BINS = 64, FRAMES = 130, MAX_BYTES = 40 * 1024 * 1024;
const form = document.querySelector('#upload-form'), fileInput = document.querySelector('#audio-file');
const button = document.querySelector('#analyze'), statusLine = document.querySelector('#status');
const player = document.querySelector('#player');
let session = null, audioURL = null;

const hzToMel = hz => hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27);
const melToHz = mel => mel < 15 ? (200 / 3) * mel : 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15));

function melFilters() {
  const points = Array.from({length: MEL_BINS + 2}, (_, i) => melToHz(hzToMel(20) + i * (hzToMel(SAMPLE_RATE / 2) - hzToMel(20)) / (MEL_BINS + 1)));
  const filters = Array.from({length: MEL_BINS}, () => new Float32Array(FFT_SIZE / 2 + 1));
  for (let m = 0; m < MEL_BINS; m++) {
    const scale = 2 / (points[m + 2] - points[m]);
    for (let k = 0; k <= FFT_SIZE / 2; k++) {
      const hz = k * SAMPLE_RATE / FFT_SIZE;
      const lower = (hz - points[m]) / (points[m + 1] - points[m]);
      const upper = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
      filters[m][k] = Math.max(0, Math.min(lower, upper)) * scale;
    }
  }
  return filters;
}
const FILTERS = melFilters();
const WINDOW = Float32Array.from({length: FFT_SIZE}, (_, n) => .5 - .5 * Math.cos(2 * Math.PI * n / FFT_SIZE));

function fft(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) [real[i], real[j], imag[i], imag[j]] = [real[j], real[i], imag[j], imag[i]];
  }
  for (let size = 2; size <= n; size <<= 1) {
    const angle = -2 * Math.PI / size;
    for (let start = 0; start < n; start += size) {
      for (let offset = 0; offset < size / 2; offset++) {
        const c = Math.cos(angle * offset), s = Math.sin(angle * offset), even = start + offset, odd = even + size / 2;
        const tr = c * real[odd] - s * imag[odd], ti = s * real[odd] + c * imag[odd];
        real[odd] = real[even] - tr; imag[odd] = imag[even] - ti;
        real[even] += tr; imag[even] += ti;
      }
    }
  }
}

function paddedSample(signal, index) {
  return index < 0 || index >= signal.length ? 0 : signal[index];
}

function spectrogram(segment) {
  const mel = new Float32Array(MEL_BINS * FRAMES), real = new Float32Array(FFT_SIZE);
  const imag = new Float32Array(FFT_SIZE), power = new Float32Array(FFT_SIZE / 2 + 1);
  let peak = 0;
  for (let frame = 0; frame < FRAMES; frame++) {
    const start = frame * HOP - FFT_SIZE / 2;
    imag.fill(0);
    for (let n = 0; n < FFT_SIZE; n++) real[n] = paddedSample(segment, start + n) * WINDOW[n];
    fft(real, imag);
    for (let k = 0; k < power.length; k++) power[k] = real[k] * real[k] + imag[k] * imag[k];
    for (let m = 0; m < MEL_BINS; m++) {
      let value = 0;
      for (let k = 0; k < power.length; k++) value += FILTERS[m][k] * power[k];
      mel[m * FRAMES + frame] = value;
      peak = Math.max(peak, value);
    }
  }
  if (peak < 1e-12) return mel.fill(-1);
  for (let i = 0; i < mel.length; i++) {
    const db = Math.max(-80, 10 * Math.log10(Math.max(1e-10, mel[i]) / peak));
    mel[i] = db / 40 + 1;
  }
  return mel;
}

async function decodeAndResample(file) {
  if (file.size > MAX_BYTES) throw new Error('Choose a file smaller than 40 MB.');
  const context = new AudioContext();
  let decoded;
  try { decoded = await context.decodeAudioData(await file.arrayBuffer()); }
  catch { throw new Error('Could not decode this audio. Try WAV, FLAC, OGG, or MP3.'); }
  finally { await context.close(); }
  if (decoded.duration < 3) throw new Error('Choose a clip at least 3 seconds long.');
  const length = Math.min(Math.ceil(decoded.duration * SAMPLE_RATE), 30 * SAMPLE_RATE);
  const offline = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate), target = mono.getChannelData(0);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const channel = decoded.getChannelData(c);
    for (let i = 0; i < channel.length; i++) target[i] += channel[i] / decoded.numberOfChannels;
  }
  const source = offline.createBufferSource(); source.buffer = mono; source.connect(offline.destination); source.start();
  const rendered = await offline.startRendering(), samples = new Float32Array(rendered.getChannelData(0));
  let peak = 0; for (const value of samples) peak = Math.max(peak, Math.abs(value));
  if (peak < 1e-6) throw new Error('This clip is silent. Choose a music clip.');
  return samples;
}

async function classify(file) {
  const samples = await decodeAndResample(file), count = Math.ceil(samples.length / SEGMENT_SAMPLES);
  const input = new Float32Array(count * MEL_BINS * FRAMES);
  for (let i = 0; i < count; i++) {
    statusLine.textContent = `Preparing excerpt ${i + 1} of ${count}…`;
    const segment = new Float32Array(SEGMENT_SAMPLES);
    segment.set(samples.subarray(i * SEGMENT_SAMPLES, (i + 1) * SEGMENT_SAMPLES));
    input.set(spectrogram(segment), i * MEL_BINS * FRAMES);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  statusLine.textContent = 'Running the CNN…';
  const tensor = new ort.Tensor('float32', input, [count, 1, MEL_BINS, FRAMES]);
  const output = (await session.run({spectrograms: tensor})).logits.data;
  const scores = new Float64Array(CLASSES.length);
  for (let row = 0; row < count; row++) {
    let max = -Infinity; for (let j = 0; j < CLASSES.length; j++) max = Math.max(max, output[row * CLASSES.length + j]);
    let sum = 0; const values = new Float64Array(CLASSES.length);
    for (let j = 0; j < CLASSES.length; j++) { values[j] = Math.exp(output[row * CLASSES.length + j] - max); sum += values[j]; }
    for (let j = 0; j < CLASSES.length; j++) scores[j] += values[j] / sum / count;
  }
  return {scores: CLASSES.map((genre, i) => ({genre, score: scores[i]})).sort((a,b) => b.score-a.score), seconds: samples.length / SAMPLE_RATE, segments: count};
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
    session = await ort.InferenceSession.create('./model.onnx', {executionProviders: ['wasm']});
    statusLine.textContent = 'Model ready. Choose a clip to begin.';
    button.disabled = !fileInput.files.length;
  } catch (error) { console.error(error); statusLine.textContent = 'The model could not load. Refresh to try again.'; }
}

fileInput.addEventListener('change', () => {
  if (audioURL) URL.revokeObjectURL(audioURL);
  const file = fileInput.files[0];
  document.querySelector('#result').hidden = true; document.querySelector('#empty').hidden = false;
  player.hidden = !file;
  if (file) { audioURL = URL.createObjectURL(file); player.src = audioURL; }
  else { player.removeAttribute('src'); player.load(); }
  button.disabled = !session || !file;
  if (session) statusLine.textContent = file ? 'Ready to analyze your clip.' : 'Choose a clip to begin.';
});

form.addEventListener('submit', async event => {
  event.preventDefault(); const file = fileInput.files[0]; if (!file || !session) return;
  button.disabled = true; fileInput.disabled = true; button.textContent = 'Analyzing…';
  try {
    const result = await classify(file); document.querySelector('#genre').textContent = result.scores[0].genre;
    const scores = document.querySelector('#scores'); scores.replaceChildren();
    result.scores.slice(0,3).forEach(item => {
      const row = document.createElement('div'); row.className = 'score';
      const label = document.createElement('div'); label.className = 'score-label';
      const name = document.createElement('span'); name.textContent = item.genre;
      const value = document.createElement('span'); value.textContent = `${(item.score * 100).toFixed(1)}%`; label.append(name,value);
      const bar = document.createElement('div'); bar.className = 'bar'; bar.setAttribute('aria-hidden','true');
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${item.score * 100}%`; bar.append(fill); row.append(label,bar); scores.append(row);
    });
    document.querySelector('#duration').textContent = `${result.seconds.toFixed(1)} seconds analyzed across ${result.segments} excerpts.`;
    document.querySelector('#empty').hidden = true; document.querySelector('#result').hidden = false; statusLine.textContent = 'Analysis complete.';
  } catch (error) { console.error(error); document.querySelector('#result').hidden = true; document.querySelector('#empty').hidden = false; statusLine.textContent = error.message || 'Could not analyze this clip.'; }
  finally { button.textContent = 'Analyze clip'; button.disabled = false; fileInput.disabled = false; }
});

window.addEventListener('load', loadModel);
