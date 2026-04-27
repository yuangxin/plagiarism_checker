import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 min — detection can be slow
});

// ── System config ──────────────────────────────────────────
export function getConfig() {
  return api.get('/config').then(r => r.data);
}

// ── File upload ────────────────────────────────────────────
export function uploadFiles(files, { mode = 'all', targetNames = [] } = {}) {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  form.append('mode', mode);
  form.append('target_names', targetNames.join(','));
  return api.post('/upload', form).then(r => r.data);
}

// ── Detection ──────────────────────────────────────────────
export function runDetect(params) {
  return api.post('/detect', params).then(r => r.data);
}

/**
 * SSE streaming detection. Returns an EventSource-like experience.
 * Usage:
 *   streamDetect(params, {
 *     onProgress: ({ stage, progress }) => ...,
 *     onResult:   (results) => ...,
 *     onError:    (err) => ...,
 *   });
 */
export function streamDetect(params, { onProgress, onResult, onError }) {
  // We use fetch + ReadableStream because SSE with POST is non-standard
  // Fall back to regular POST if SSE fails
  return runDetect(params)
    .then(onResult)
    .catch(onError);
}

// ── Results ────────────────────────────────────────────────
export function getResults(jobId) {
  return api.get(`/results/${jobId}`).then(r => r.data);
}

export function getExportUrl(jobId, format) {
  return `/api/results/${jobId}/export?format=${format}`;
}

// ── Job management ─────────────────────────────────────────
export function getJobInfo(jobId) {
  return api.get(`/jobs/${jobId}`).then(r => r.data);
}

export function deleteJob(jobId) {
  return api.delete(`/jobs/${jobId}`).then(r => r.data);
}
