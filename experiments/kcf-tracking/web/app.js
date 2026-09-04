const frameWidth = __WIDTH__;
const frameHeight = __HEIGHT__;
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const targetPreview = document.getElementById('targetPreview');
const targetPlaceholder = document.getElementById('targetPlaceholder');
let selecting = false;
let startPoint = null;
let currentPoint = null;
let latest = {};
let lastTargetSequence = -1;

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
  };
}

function normalizedBox(a, b) {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return { x, y, w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
}

function boxValues(box) {
  if (!box) return null;
  if (Array.isArray(box)) return { x: box[0], y: box[1], w: box[2], h: box[3] };
  return box;
}

function drawBox(box, color, lineWidth, dashed = false) {
  const values = boxValues(box);
  if (!values) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dashed ? [8, 5] : []);
  ctx.strokeRect(values.x * frameWidth, values.y * frameHeight, values.w * frameWidth, values.h * frameHeight);
  ctx.restore();
}

function drawTrackingCoordinates() {
  const values = boxValues(latest.bbox);
  if (!values) return;
  const x = Math.round(values.x * frameWidth);
  const y = Math.round(values.y * frameHeight);
  const w = Math.round(values.w * frameWidth);
  const h = Math.round(values.h * frameHeight);
  const centerX = Math.round(x + w / 2);
  const centerY = Math.round(y + h / 2);
  const recovering = Boolean(latest.recovering);
  const state = latest.tracking ? 'KCF TRACKING' : (recovering ? 'KCF RECOVERING' : 'KCF LOST (last position)');
  const lines = [
    state,
    `bbox x=${x} y=${y} w=${w} h=${h}`,
    `center=(${centerX}, ${centerY}) px`,
  ];
  const score = Number.isFinite(latest.recovery_score) ? latest.recovery_score : latest.match_score;
  if (Number.isFinite(score)) {
    lines.push(`${recovering ? 'recovery' : 'match'} score=${score.toFixed(2)}`);
  }
  ctx.save();
  ctx.font = 'bold 15px system-ui, sans-serif';
  const lineHeight = 20;
  const boxWidth = Math.max(...lines.map(line => ctx.measureText(line).width)) + 16;
  const boxHeight = lines.length * lineHeight + 10;
  ctx.fillStyle = latest.tracking ? 'rgba(20, 83, 45, .88)' : (recovering ? 'rgba(113, 63, 18, .9)' : 'rgba(127, 29, 29, .88)');
  ctx.fillRect(8, 8, boxWidth, boxHeight);
  ctx.fillStyle = '#ffffff';
  lines.forEach((line, index) => ctx.fillText(line, 16, 26 + index * lineHeight));
  ctx.strokeStyle = latest.tracking ? '#86efac' : (recovering ? '#facc15' : '#fca5a5');
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'rgba(250, 204, 21, .78)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(frameWidth / 2, 0); ctx.lineTo(frameWidth / 2, frameHeight);
  ctx.moveTo(0, frameHeight / 2); ctx.lineTo(frameWidth, frameHeight / 2);
  ctx.stroke();
  drawBox(latest.roi, '#38bdf8', 2, true);
  drawBox(latest.bbox, latest.tracking ? '#86efac' : (latest.recovering ? '#facc15' : '#fca5a5'), 3, !latest.tracking);
  drawTrackingCoordinates();
  if (selecting && startPoint && currentPoint) drawBox(normalizedBox(startPoint, currentPoint), '#facc15', 2);
}

function updateTargetPreview(data) {
  const hasTarget = Boolean(data.roi && data.target_sequence > 0);
  targetPreview.style.display = hasTarget ? 'block' : 'none';
  targetPlaceholder.style.display = hasTarget ? 'none' : 'flex';
  if (!hasTarget) {
    lastTargetSequence = -1;
    targetPreview.removeAttribute('src');
    return;
  }
  if (data.target_sequence !== lastTargetSequence) {
    lastTargetSequence = data.target_sequence;
    targetPreview.src = `/target.jpg?seq=${data.target_sequence}`;
  }
}

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body), cache: 'no-store'
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  latest = data;
  drawOverlay();
  return data;
}

function updateStatus(data) {
  latest = data;
  document.getElementById('tracker').textContent = data.tracker || '--';
  const tracking = document.getElementById('tracking');
  tracking.textContent = data.tracking ? '追跡中' : (data.recovering ? '画面内を再捕捉中' : (data.roi ? '見失い' : '未選択'));
  tracking.className = data.tracking ? 'ok' : (data.roi ? 'warn' : '');
  document.getElementById('pan').textContent = data.pan_angle ?? '--';
  document.getElementById('tilt').textContent = data.tilt_angle ?? '--';
  document.getElementById('fps').textContent = data.tracking_fps?.toFixed?.(1) ?? '--';
  document.getElementById('message').textContent = data.message || '接続正常';
  drawOverlay();
  updateTargetPreview(data);
}

canvas.addEventListener('pointerdown', event => {
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  selecting = true;
  startPoint = pointFromEvent(event);
  currentPoint = startPoint;
  drawOverlay();
});
canvas.addEventListener('pointermove', event => {
  if (!selecting) return;
  currentPoint = pointFromEvent(event);
  drawOverlay();
});
canvas.addEventListener('pointerup', async event => {
  if (!selecting) return;
  currentPoint = pointFromEvent(event);
  const box = normalizedBox(startPoint, currentPoint);
  selecting = false;
  startPoint = null;
  currentPoint = null;
  drawOverlay();
  if (box.w < 0.03 || box.h < 0.03) {
    document.getElementById('message').textContent = '選択範囲が小さすぎます';
    return;
  }
  try { updateStatus(await post('/api/roi', box)); }
  catch (error) { document.getElementById('message').textContent = error.message; }
});

document.getElementById('stop').addEventListener('click', () => post('/api/stop').then(updateStatus).catch(console.error));
document.getElementById('center').addEventListener('click', () => post('/api/center').then(updateStatus).catch(console.error));
document.getElementById('release').addEventListener('click', () => post('/api/release').then(updateStatus).catch(console.error));

async function refresh() {
  try { updateStatus(await fetch('/api/status', {cache: 'no-store'}).then(r => r.json())); }
  catch (error) { document.getElementById('message').textContent = error.message; }
}
setInterval(refresh, 150);
refresh();
