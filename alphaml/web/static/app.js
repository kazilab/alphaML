'use strict';

/* ------------------------------------------------------------------ helpers */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  options: null,
  datasets: [],
  jobs: [],
  view: 'train',
  jobId: null,
  logOffset: 0,
  detailTimer: null,
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function errorText(detail) {
  if (Array.isArray(detail)) return detail.map(d => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join('; ');
  return detail || 'Request failed';
}

class LoginRequired extends Error {
  constructor() {
    super('Please log in.');
    this.silent = true;  // the login screen is shown instead of an error message
  }
}

// Every request carries this header: the server refuses state-changing requests without it, so other websites
// cannot make the browser upload data or start jobs here.
const API_HEADERS = { 'X-AlphaML': '1' };

function loginExpired(path, status) {
  return status === 401 && state.options && state.options.auth && !path.startsWith('/api/auth/');
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...API_HEADERS, ...(options.headers || {}) } });
  if (loginExpired(path, response.status)) {
    showAuth();
    throw new LoginRequired();
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = errorText((await response.json()).detail); } catch (_) { /* not JSON */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

// Files are sent in chunks: a public proxy such as Cloudflare limits one request to 100 MB.
// Returns the upload id; onBytes(n) is called with the number of bytes the server has stored.
async function uploadInChunks(file, onBytes) {
  const max = state.options.max_upload_bytes;
  if (max && file.size > max) throw new Error(`${file.name} is larger than ${fmtSize(max)}.`);
  const upload = await postJSON('/api/uploads', { filename: file.name, size: file.size });
  let offset = 0;
  try {
    while (offset < file.size) {
      offset = await putChunk(upload.id, offset, file.slice(offset, offset + upload.chunk_size));
      onBytes(offset);
    }
  } catch (err) {
    api(`/api/uploads/${upload.id}`, { method: 'DELETE' }).catch(() => {});
    throw err;
  }
  return upload.id;
}

async function putChunk(uploadId, offset, chunk, attempts = 4) {
  for (let attempt = 1; ; attempt++) {
    try {
      const reply = await api(`/api/uploads/${uploadId}?offset=${offset}`,
        { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: chunk });
      return reply.received;
    } catch (err) {
      // Retry network errors and server-side failures, not refused chunks
      const temporary = !err.status || err.status >= 500 || err.status === 429;
      if (err.silent || !temporary || attempt >= attempts) throw err;
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      try {
        // The chunk may have arrived although the reply was lost: continue from what the server has
        const status = await api(`/api/uploads/${uploadId}`);
        if (status.received > offset) return status.received;
      } catch (_) { /* retry the chunk */ }
    }
  }
}

function uploadingGuard(on) {
  // Leaving the page would stop an upload in progress
  window.onbeforeunload = on ? () => 'An upload is in progress.' : null;
}

function postJSON(path, body) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

function toast(message) {
  const box = $('#toast');
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 3500);
}

function report(err) {
  if (!err.silent) toast(err.message);
}

function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleDateString(undefined, { dateStyle: 'medium' }) : '—';
}

function fillSelect(select, values, selected, label = v => v) {
  select.replaceChildren(...values.map(v => el('option', { value: v, text: label(v) })));
  if (selected !== undefined && values.includes(selected)) select.value = selected;
}

function fmt(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return value.toLocaleString();
    return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3);
  }
  return String(value);
}

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1 << 20) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1 << 20)).toFixed(1)} MB`;
}

function fmtTime(iso) {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';
}

function duration(job) {
  if (!job.started) return null;
  const end = job.finished ? new Date(job.finished) : new Date();
  const s = Math.max(0, Math.round((end - new Date(job.started)) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`;
}

const isActive = job => job.status === 'queued' || job.status === 'running';
const fileUrl = (jobId, name, download) =>
  `/api/jobs/${jobId}/files/${name.split('/').map(encodeURIComponent).join('/')}${download ? '?download=true' : ''}`;

/* ------------------------------------------------------------------ routing */

function route() {
  const [, view = 'train', id = null] = location.hash.split('/');
  state.view = ['train', 'predict', 'jobs'].includes(view) ? view : 'train';
  for (const section of $$('.view')) section.hidden = section.id !== `view-${state.view}`;
  for (const tab of $$('.tabs a')) tab.classList.toggle('active', tab.dataset.tab === state.view);
  stopDetailPolling();
  if (state.view === 'jobs') {
    loadJobs().then(() => {
      const target = id || (state.jobs[0] && state.jobs[0].id);
      if (target) openJob(target); else renderEmptyDetail();
    });
  } else if (state.view === 'predict') {
    loadJobs().then(refreshModelJobs);
  }
}

/* ------------------------------------------------------------------ train */

const trainForm = () => $('#train-form');

function initTrainForm() {
  const o = state.options;
  const d = o.train_defaults;
  const form = trainForm();
  fillSelect(form.model_name, o.algorithms, 'XGBoost');
  fillSelect(form.automl_metric, Object.keys(o.rank_metrics), d.automl_metric, v => o.rank_metrics[v]);
  $('#automl-models').replaceChildren(...o.algorithms.map(name => el('label', { class: 'chip' },
    el('input', { type: 'checkbox', name: 'automl_models', value: name, checked: d.automl_models.includes(name) || null }),
    el('span', { text: name }))));
  $('#automl-all').addEventListener('click', () => setAlgorithms(true));
  $('#automl-none').addEventListener('click', () => setAlgorithms(false));
  form.mode.value = d.model_name === 'AutoML' ? 'automl' : 'single';
  fillSelect(form.param_search, o.param_search_methods, d.param_search);
  fillSelect(form.sampling_method, o.sampling_methods, d.sampling_method, v => ({ no: 'None', over: 'Oversample (SMOTE)', under: 'Undersample (Tomek links + random)' })[v] || v);
  fillSelect(form.normalization, o.normalization_methods, d.normalization, v => ({ min_max: 'Min-max per sample', standardization: 'Standardize per sample', None: 'None' })[v] || v);
  fillSelect(form.log_type, o.log_types, d.log_type, v => v === 'None' ? 'Not log-transformed' : `${v}(x+1)`);
  for (const name of ['features', 'features2', 'features3', 'features4']) {
    fillSelect(form[name], o.feature_methods, d[name]);
  }
  for (const [name, value] of Object.entries(d)) {
    const input = form.elements[name];
    if (!input || !input.tagName || input.tagName === 'SELECT') continue;  // selects and chip groups are set above
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = value;
  }
  form.addEventListener('change', updateTrainForm);
  form.addEventListener('submit', submitTrain);
  $('#dataset-select').addEventListener('change', () => { updateDatasetInfo(); updateTrainForm(); });
  $('#col_label').addEventListener('change', updateClassSelects);
  $('#pos_class').addEventListener('change', updateBalance);
  $('#neg_class').addEventListener('change', updateBalance);
  $('#dataset-upload-toggle').addEventListener('click', () => { $('#dataset-upload').hidden = false; $('#up-name').focus(); });
  $('#dataset-upload-cancel').addEventListener('click', () => { $('#dataset-upload').hidden = true; });
  $('#dataset-upload-submit').addEventListener('click', uploadDataset);
  $('#dataset-example').addEventListener('click', loadExampleDataset);
  $('#dataset-delete').addEventListener('click', deleteDataset);
  updateTrainForm();
}

function setAlgorithms(checked) {
  for (const box of $$('input[name="automl_models"]')) box.checked = checked;
  updateTrainForm();
}

function selectedAlgorithms() {
  return $$('input[name="automl_models"]:checked').map(box => box.value);
}

function currentDataset() {
  return state.datasets.find(d => d.id === $('#dataset-select').value) || null;
}

async function loadDatasets(selectId) {
  state.datasets = await api('/api/datasets');
  const select = $('#dataset-select');
  const previous = selectId || select.value;
  if (state.datasets.length) {
    select.replaceChildren(...state.datasets.map(d =>
      el('option', { value: d.id, text: d.public && state.options.auth ? `${d.name} (shared)` : d.name })));
    if (state.datasets.some(d => d.id === previous)) select.value = previous;
  } else {
    select.replaceChildren(el('option', { value: '', text: 'No data sets yet: upload one or load the example' }));
  }
  updateDatasetInfo();
  updateTrainForm();
}

function updateDatasetInfo() {
  const ds = currentDataset();
  const summary = $('#dataset-summary');
  const colSelect = $('#col_label');
  // The shared example data set cannot be deleted by users
  $('#dataset-delete').hidden = !ds || (state.options.auth && ds.public);
  if (!ds) {
    summary.textContent = '';
    for (const s of [colSelect, $('#pos_class'), $('#neg_class')]) s.replaceChildren();
    $('#class-balance').textContent = '';
    return;
  }
  const extras = ['data_for_feature_selection', 'unlabeled_data', 'suggested_features']
    .filter(k => ds.files[k]).map(k => k.replaceAll('_', ' '));
  summary.textContent = `${fmt(ds.n_samples)} samples × ${fmt(ds.n_features)} features · ` +
    `${fmt(ds.n_matched_labels)} samples have labels` + (extras.length ? ` · also: ${extras.join(', ')}` : '') +
    (ds.expires ? ` · deleted on ${fmtDate(ds.expires)}` : '');
  const columns = ds.label_columns.map(c => c.name);
  const preferred = columns.includes(state.options.train_defaults.col_label)
    ? state.options.train_defaults.col_label : columns[0];
  fillSelect(colSelect, columns, colSelect.value && columns.includes(colSelect.value) ? colSelect.value : preferred);
  updateClassSelects();
}

function updateClassSelects() {
  const ds = currentDataset();
  if (!ds) return;
  const column = ds.label_columns.find(c => c.name === $('#col_label').value);
  const values = column ? column.values.map(v => v.value) : [];
  const d = state.options.train_defaults;
  const pick = (want, fallback) => values.find(v => v.toLowerCase() === want.toLowerCase()) || fallback;
  const pos = pick(d.pos_class, values[0]);
  const neg = pick(d.neg_class, values.find(v => v !== pos));
  fillSelect($('#pos_class'), values, pos);
  fillSelect($('#neg_class'), values, neg);
  updateBalance();
}

function updateBalance() {
  const ds = currentDataset();
  const column = ds && ds.label_columns.find(c => c.name === $('#col_label').value);
  if (!column) return;
  const count = v => (column.values.find(x => x.value === v) || { count: 0 }).count;
  const pos = $('#pos_class').value;
  const neg = $('#neg_class').value;
  const others = column.values.filter(v => v.value !== pos && v.value !== neg).reduce((a, v) => a + v.count, 0);
  $('#class-balance').textContent = pos === neg
    ? 'Choose two different classes.'
    : `${count(pos)} ${pos} vs ${count(neg)} ${neg}` +
      (others ? ` · ${others} ${others === 1 ? 'sample' : 'samples'} with other labels left out` : '');
}

function updateTrainForm() {
  const form = trainForm();
  const methods = $$('.fs-method', form).map(s => s.value);
  for (const field of $$('.fs-params [data-for]', form)) field.hidden = !methods.includes(field.dataset.for);

  const automl = form.mode.value === 'automl';
  for (const block of $$('[data-mode]', form)) block.hidden = block.dataset.mode !== form.mode.value;
  const search = automl ? 'Optuna' : form.param_search.value;
  const roc = form.run_roc.checked;
  form.n_trials.disabled = !(search === 'Optuna' || search === 'Bayes' || roc);
  form.cv.disabled = !(search !== 'Predefined' || roc);
  $('#trials-label').textContent = automl ? 'Optuna trials per algorithm' : 'Trials / iterations';
  $('[data-search="optuna"]', form).hidden = search !== 'Optuna';

  const warnings = [];
  const ds = currentDataset();
  if (methods.includes('Suggested') && ds && !ds.files.suggested_features) {
    warnings.push('"Suggested" needs a suggested features file in the data set.');
  }
  const slow = ['SeqFeatSel', 'RecursiveFeatElim'].filter(m => methods.indexOf(m) === 0);
  if (slow.length && ds && ds.n_features > 2000) {
    warnings.push(`${slow[0]} on ${fmt(ds.n_features)} features can take hours: put a fast method (e.g. HVF) first.`);
  }
  $('#fs-warning').hidden = !warnings.length;
  $('#fs-warning').textContent = warnings.join(' ');
}

async function uploadDataset() {
  const keys = ['labeled_data', 'data_labels', 'data_for_feature_selection', 'unlabeled_data', 'suggested_features'];
  const chosen = keys.map(key => [key, $(`#up-${key}`).files[0]]).filter(([, file]) => file);
  const name = $('#up-name').value.trim() || ($('#up-labeled_data').files[0] || {}).name || '';
  if (!$('#up-labeled_data').files.length || !$('#up-data_labels').files.length) {
    toast('Choose the labeled data and labels files.');
    return;
  }
  const button = $('#dataset-upload-submit');
  const bar = $('#up-progress');
  const status = $('#up-status');
  const total = chosen.reduce((sum, [, file]) => sum + file.size, 0) || 1;
  const uploaded = {};
  let done = 0;
  button.disabled = true;
  bar.hidden = false;
  bar.value = 0;
  uploadingGuard(true);
  try {
    for (const [key, file] of chosen) {
      uploaded[key] = await uploadInChunks(file, sent => {
        bar.value = Math.round(100 * (done + sent) / total);
        status.textContent = `${fmtSize(done + sent)} of ${fmtSize(total)}`;
      });
      done += file.size;
    }
    status.textContent = 'Reading the data set…';
    const record = await postJSON('/api/datasets/from-uploads', { name, files: uploaded });
    $('#dataset-upload').hidden = true;
    for (const key of keys) $(`#up-${key}`).value = '';
    $('#up-name').value = '';
    await loadDatasets(record.id);
    toast(`Uploaded "${record.name}"`);
  } catch (err) {
    for (const id of Object.values(uploaded)) api(`/api/uploads/${id}`, { method: 'DELETE' }).catch(() => {});
    report(err);
  } finally {
    uploadingGuard(false);
    button.disabled = false;
    bar.hidden = true;
    bar.value = 0;
    status.textContent = '';
  }
}

async function loadExampleDataset() {
  const button = $('#dataset-example');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Downloading…';
  try {
    const record = await api('/api/datasets/example', { method: 'POST' });
    await loadDatasets(record.id);
    toast('Example data loaded');
  } catch (err) {
    report(err);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function deleteDataset() {
  const ds = currentDataset();
  if (!ds || !confirm(`Delete data set "${ds.name}"? Job results are kept.`)) return;
  try {
    await api(`/api/datasets/${ds.id}`, { method: 'DELETE' });
    await loadDatasets();
    toast('Data set deleted');
  } catch (err) {
    report(err);
  }
}

function collectTrainParams() {
  const form = trainForm();
  const params = {};
  const skip = ['mode', 'automl_models'];
  for (const input of form.elements) {
    if (!input.name || skip.includes(input.name) || ['file', 'button', 'submit'].includes(input.type)) continue;
    if (input.type === 'checkbox') params[input.name] = input.checked;
    else if (input.type === 'number') params[input.name] = input.value === '' ? null : Number(input.value);
    else params[input.name] = input.value;
  }
  // Disabled fields still carry their (default) values: the API needs them for ROC repeats
  if (form.mode.value === 'automl') {
    params.model_name = 'AutoML';
    params.automl_models = selectedAlgorithms();
  }
  return params;
}

async function submitTrain(event) {
  event.preventDefault();
  const errorBox = $('#train-error');
  errorBox.hidden = true;
  const params = collectTrainParams();
  const problem = !params.dataset_id ? 'Choose or upload a data set.'
    : !params.col_label ? 'Choose a class column.'
    : params.pos_class === params.neg_class ? 'The positive and negative classes must differ.'
    : params.model_name === 'AutoML' && !params.automl_models.length ? 'Select at least one algorithm for AutoML.'
    : null;
  if (problem) {
    errorBox.textContent = problem;
    errorBox.hidden = false;
    return;
  }
  const button = $('#train-submit');
  button.disabled = true;
  try {
    const job = await postJSON('/api/jobs/train', params);
    location.hash = `#/jobs/${job.id}`;
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ predict */

function initPredictForm() {
  const form = $('#predict-form');
  for (const value of state.options.normalization_methods) {
    form.normalization.append(el('option', { value, text: value }));
  }
  $('#model-upload-option').hidden = !state.options.allow_model_upload;
  for (const radio of $$('input[name="model_source"]', form)) {
    radio.addEventListener('change', () => {
      for (const block of $$('[data-source]', form)) block.hidden = block.dataset.source !== form.model_source.value;
    });
  }
  form.addEventListener('submit', submitPredict);
}

function refreshModelJobs(selectId) {
  const trained = state.jobs.filter(j => j.kind === 'train' && j.status === 'succeeded');
  const select = $('#model-job');
  const previous = selectId || select.value;
  select.replaceChildren(...trained.map(j => el('option', { value: j.id, text: `${j.title} · ${fmtTime(j.finished)}` })));
  if (trained.some(j => j.id === previous)) select.value = previous;
  select.disabled = !trained.length;
  $('#model-job-empty').hidden = trained.length > 0;
}

async function submitPredict(event) {
  event.preventDefault();
  const form = $('#predict-form');
  const errorBox = $('#predict-error');
  errorBox.hidden = true;
  const fail = message => { errorBox.textContent = message; errorBox.hidden = false; };
  const testFile = $('#test-file').files[0];
  if (!testFile) return fail('Choose a test data file.');

  const fromJob = form.model_source.value === 'job';
  const modelFile = fromJob ? null : $('#model-file').files[0];
  if (fromJob && !$('#model-job').value) return fail('Choose a training job, or upload a model file.');
  if (!fromJob && !modelFile) return fail('Choose a model bundle file.');

  const button = $('#predict-submit');
  const label = button.textContent;
  const files = [testFile, modelFile].filter(Boolean);
  const total = files.reduce((sum, file) => sum + file.size, 0) || 1;
  const uploaded = [];
  let done = 0;
  button.disabled = true;
  uploadingGuard(true);
  try {
    for (const file of files) {
      uploaded.push(await uploadInChunks(file, sent => {
        button.textContent = `Uploading ${Math.round(100 * (done + sent) / total)}%…`;
      }));
      done += file.size;
    }
    const data = new FormData();
    data.append('test_data_upload', uploaded[0]);
    if (fromJob) data.append('model_job_id', $('#model-job').value);
    else data.append('model_upload', uploaded[1]);
    for (const name of ['normalization', 'col_label', 'pos_class', 'neg_class']) {
      if (form[name].value.trim()) data.append(name, form[name].value.trim());
    }
    data.append('run_shap', form.run_shap.checked);
    data.append('run_lime', form.run_lime.checked);
    const job = await api('/api/jobs/predict', { method: 'POST', body: data });
    location.hash = `#/jobs/${job.id}`;
  } catch (err) {
    for (const id of uploaded) api(`/api/uploads/${id}`, { method: 'DELETE' }).catch(() => {});
    fail(err.message);
  } finally {
    uploadingGuard(false);
    button.disabled = false;
    button.textContent = label;
  }
}

/* ------------------------------------------------------------------ jobs */

async function loadJobs() {
  try {
    state.jobs = await api('/api/jobs');
  } catch (err) {
    report(err);
    return;
  }
  const active = state.jobs.filter(isActive).length;
  $('#active-count').hidden = !active;
  $('#active-count').textContent = active;
  renderJobList();
}

function renderJobList() {
  const list = $('#job-list');
  list.replaceChildren(...state.jobs.map(job => el('li', {},
    el('a', { href: `#/jobs/${job.id}`, class: job.id === state.jobId ? 'active' : null },
      el('span', { class: `dot ${job.status}`, title: job.status }),
      el('span', { class: 'title', text: job.title }),
      el('span', { class: 'meta', text: [job.status, fmtTime(job.created), duration(job)].filter(Boolean).join(' · ') })))));
  $('#job-list-empty').hidden = state.jobs.length > 0;
}

function renderEmptyDetail() {
  $('#job-detail').replaceChildren(el('p', { class: 'muted', text: 'No jobs yet. Start one from Train or Predict.' }));
}

function stopDetailPolling() {
  clearTimeout(state.detailTimer);
  state.detailTimer = null;
}

async function openJob(jobId) {
  if (state.jobId !== jobId) {
    state.jobId = jobId;
    state.logOffset = 0;
  }
  renderJobList();
  stopDetailPolling();
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    $('#job-detail').replaceChildren(el('p', { class: 'error', text: err.message }));
    return;
  }
  if (state.view !== 'jobs' || state.jobId !== jobId) return;
  renderJobDetail(job);
  if (isActive(job)) {
    state.detailTimer = setTimeout(async () => {
      await loadJobs();
      if (state.view === 'jobs' && state.jobId === jobId) openJob(jobId);
    }, 2000);
  }
}

function renderJobDetail(job) {
  const root = $('#job-detail');
  const logBox = $('#job-log', root);
  const keepLog = logBox && logBox.dataset.job === job.id ? logBox : null;
  const p = job.params || {};
  const meta = [
    job.kind === 'train' ? 'Training' : 'Prediction',
    p.dataset_name ? `data: ${p.dataset_name}` : null,
    `created ${fmtTime(job.created)}`,
    duration(job) ? `runtime ${duration(job)}` : null,
    job.expires ? `deleted on ${fmtDate(job.expires)}` : null,
  ].filter(Boolean).join(' · ');

  const actions = el('div', { class: 'job-actions' },
    isActive(job) && el('button', { class: 'btn danger', text: 'Cancel', onclick: () => jobAction(job, 'cancel') }),
    job.kind === 'train' && job.status === 'succeeded' &&
      el('button', { class: 'btn', text: 'Predict with this model', onclick: () => usePredictModel(job.id) }),
    !isActive(job) && job.files && job.files.length &&
      el('a', { class: 'btn', href: `/api/jobs/${job.id}/archive`, text: 'Download all' }),
    !isActive(job) && el('button', { class: 'btn ghost danger', text: 'Delete', onclick: () => jobAction(job, 'delete') }));

  const parts = [
    el('div', { class: 'job-head' },
      el('div', {}, el('h1', {}, job.title, el('span', { class: `badge ${job.status}`, text: job.status })),
        el('p', { class: 'muted small', text: meta })),
      actions),
  ];
  if (isActive(job)) {
    const progress = job.progress || {};
    parts.push(el('p', { class: 'progress-line' },
      el('strong', { text: job.status === 'queued' ? 'Waiting for a free worker' : (progress.stage || 'Starting') }),
      progress.detail ? ` — ${progress.detail}` : ''));
  }
  if (job.error) parts.push(el('p', { class: 'error-box', text: job.error }));
  if (job.status === 'succeeded' && job.summary) parts.push(...renderSummary(job));

  const images = (job.files || []).filter(f => f.kind === 'image');
  if (images.length) {
    parts.push(el('h3', { text: 'Plots' }), el('div', { class: 'gallery' }, images.map(f => el('figure', {},
      el('img', { src: fileUrl(job.id, f.name), alt: f.name, loading: 'lazy', onclick: () => lightbox(fileUrl(job.id, f.name)) }),
      el('figcaption', { text: prettyName(f.name) })))));
  }
  if (job.files && job.files.length) parts.push(...renderFiles(job));
  parts.push(renderParams(job));

  const log = keepLog || el('pre', { class: 'log', id: 'job-log', 'data-job': job.id });
  const logOpen = keepLog ? keepLog.closest('details').open : isActive(job) || job.status === 'failed';
  parts.push(el('details', { class: 'params', open: logOpen || null }, el('summary', { text: 'Log' }), log));
  root.replaceChildren(...parts);
  refreshLog(job.id, log);
}

function prettyName(name) {
  return name.replace(/\.(png|pdf)$/, '').replace(/^.*?_(fit_(Yes|No)_)/, '').replaceAll('_', ' ');
}

function renderSummary(job) {
  const s = job.summary;
  const parts = [];
  if (job.kind === 'predict') {
    const tiles = [['Samples', s.n_samples], ['Features', s.n_features], ...Object.entries(s.class_counts || {})];
    parts.push(statTiles(tiles));
    if (s.predictions && s.predictions.length) {
      parts.push(el('h3', { text: s.predictions.length < s.n_samples ? `Predictions (first ${s.predictions.length})` : 'Predictions' }),
        dataTable(Object.keys(s.predictions[0]), s.predictions.map(r => Object.values(r))));
    }
    return parts;
  }
  const scores = Object.fromEntries((s.scores || []).map(r => [r.Metric, r]));
  const test = name => scores[name] ? scores[name].Test : null;
  if (s.selected_model) {
    parts.push(el('p', { class: 'selected-model' }, 'AutoML selected ', el('strong', { text: s.selected_model }),
      s.ensemble_members ? ` (soft-voting ensemble of ${s.ensemble_members.join(' + ')})` : '',
      ` out of ${new Set(s.leaderboard.map(r => r.Model).filter(m => m !== 'Ensemble')).size} algorithms.`));
  }
  parts.push(statTiles([
    ['NegLog2-RMSL', s.pscore],
    ['Test accuracy', test('Accuracy')],
    ['Test AUC', test('AUC (probability)') ?? test('AUC')],
    ['Test MCC', test('MCC')],
    s.roc_auc !== undefined ? ['ROC AUC (repeated CV)', s.roc_auc] : null,
    ['Features', s.n_features],
    ['Train / test samples', `${s.n_train} / ${s.n_test}`],
  ].filter(Boolean)));
  if (s.leaderboard) {
    const metric = state.options.rank_metrics[s.automl_metric] || 'Score';
    const cols = ['Rank', 'Model', 'Stage', 'Score', 'SD', 'Train-CV gap', 'AUC', 'Accuracy', 'MCC', 'Seconds'];
    parts.push(el('h3', { text: 'AutoML leaderboard' }),
      el('p', { class: 'small muted', text: `Score: ${metric}, mean over cross-validation folds of the training data. ` +
        'The test set was not used to choose the model.' }),
      dataTable(cols, s.leaderboard.map(r => cols.map(c => r[c])), 1, { 'Train-CV gap': 'Gap', Seconds: 'Time (s)' }),
      el('p', { class: 'small muted', text: 'Gap: training score minus cross-validation score (overfitting).' }));
  }
  if (s.scores) {
    const cols = ['Metric', 'Train', 'Test', 'Test loss'];
    parts.push(el('h3', { text: 'Test set scores' }),
      dataTable(cols, s.scores.map(r => cols.map(c => r[c])), 'NegLog2-RMSL'));
  }
  return parts;
}

function statTiles(items) {
  return el('div', { class: 'stats' }, items.map(([label, value]) =>
    el('div', { class: 'stat' }, el('div', { class: 'label', text: label }), el('div', { class: 'value', text: fmt(value) }))));
}

function dataTable(columns, rows, highlight, labels = {}) {
  return el('div', { class: rows.length > 25 ? 'table-wrap' : 'table-wrap full' }, el('table', {},
    el('thead', {}, el('tr', {}, columns.map(c => el('th', { text: labels[c] || c })))),
    el('tbody', {}, rows.map(row => el('tr', { class: highlight && row[0] === highlight ? 'highlight' : null },
      row.map(v => el('td', { class: typeof v === 'number' ? 'num' : null, text: fmt(v) })))))));
}

function renderFiles(job) {
  const preview = el('div', { id: 'file-preview' });
  const rows = job.files.map(f => el('tr', {},
    el('td', {}, f.kind === 'pdf' || f.kind === 'image'
      ? el('a', { href: fileUrl(job.id, f.name), target: '_blank', rel: 'noopener', text: f.name })
      : f.name),
    el('td', { class: 'kind', text: f.kind }),
    el('td', { class: 'num', text: fmtSize(f.size) }),
    el('td', {},
      /\.(csv|xlsx)$/i.test(f.name) && el('button', { class: 'link-btn', type: 'button', text: 'Preview', onclick: () => previewFile(job.id, f.name, preview) }),
      ' ',
      el('a', { href: fileUrl(job.id, f.name, true), text: 'Download' }))));
  return [el('h3', { text: `Files (${job.files.length})` }),
    el('div', { class: 'table-wrap files' }, el('table', {},
      el('thead', {}, el('tr', {}, el('th', { text: 'Name' }), el('th', { text: 'Type' }), el('th', { text: 'Size' }), el('th', {}))),
      el('tbody', {}, rows))),
    preview];
}

async function previewFile(jobId, name, container) {
  try {
    const data = await api(`/api/jobs/${jobId}/preview/${name.split('/').map(encodeURIComponent).join('/')}`);
    container.replaceChildren(el('h3', { text: `Preview: ${name}` }), dataTable(data.columns, data.rows));
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    report(err);
  }
}

function renderParams(job) {
  const p = job.params || {};
  const values = { ...(p.train || p.predict || {}) };
  if (job.summary && job.summary.parameters) {
    for (const [k, v] of Object.entries(job.summary.parameters)) values[`model: ${k}`] = v;
  }
  return el('details', { class: 'params' }, el('summary', { text: 'Settings' }),
    el('div', { class: 'kv' }, Object.entries(values).map(([k, v]) =>
      el('div', {}, el('span', { text: k }), el('span', { text: v === null ? 'default' : (typeof v === 'object' ? JSON.stringify(v) : fmt(v)) })))));
}

async function refreshLog(jobId, box) {
  try {
    const data = await api(`/api/jobs/${jobId}/log?offset=${state.logOffset}`);
    if (state.jobId !== jobId) return;
    if (data.text) {
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      box.append(data.text);
      state.logOffset = data.offset;
      if (atBottom) box.scrollTop = box.scrollHeight;
    } else if (!box.textContent) {
      box.textContent = '';
    }
  } catch (_) { /* log not available yet */ }
}

async function jobAction(job, action) {
  if (action === 'delete' && !confirm(`Delete "${job.title}" and its results?`)) return;
  try {
    if (action === 'cancel') {
      await api(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
      toast('Job cancelled');
    } else {
      await api(`/api/jobs/${job.id}`, { method: 'DELETE' });
      state.jobId = null;
      toast('Job deleted');
    }
    await loadJobs();
    if (action === 'cancel') openJob(job.id);
    else location.hash = '#/jobs';
  } catch (err) {
    report(err);
  }
}

function usePredictModel(jobId) {
  location.hash = '#/predict';
  setTimeout(() => {
    const form = $('#predict-form');
    form.model_source.value = 'job';
    for (const block of $$('[data-source]', form)) block.hidden = block.dataset.source !== 'job';
    refreshModelJobs(jobId);
  }, 0);
}

function lightbox(src) {
  const box = el('div', { class: 'lightbox', onclick: () => box.remove() }, el('img', { src, alt: '' }));
  document.addEventListener('keydown', function close(e) {
    if (e.key === 'Escape') { box.remove(); document.removeEventListener('keydown', close); }
  });
  document.body.append(box);
}

/* ------------------------------------------------------------------ accounts */

function authMode() {
  return $('input[name="auth_mode"]:checked').value;
}

function updateAuthForm() {
  const register = authMode() === 'register';
  for (const field of $$('[data-auth="register"]')) field.hidden = !register;
  for (const field of $$('[data-auth="login"]')) field.hidden = register;
  $('#signup-code-field').hidden = !register || !state.options.signup_code_required;
  $('#auth-form').password.autocomplete = register ? 'new-password' : 'current-password';
  $('#auth-submit').textContent = register ? 'Create account' : 'Log in';
  $('#auth-error').hidden = true;
}

function retentionText() {
  const days = state.options.retention_days;
  return days ? `Data sets and results are deleted automatically ${days} days after they are uploaded or ` +
    'submitted. Download the results you want to keep.' : '';
}

function showAuth() {
  stopDetailPolling();
  state.loggedIn = false;
  state.jobs = [];
  state.datasets = [];
  $('#job-list').replaceChildren();
  renderEmptyDetail();
  $('#app-main').hidden = true;
  $('#tabs').hidden = true;
  $('#user-menu').hidden = true;
  $('#view-privacy').hidden = true;
  $('#view-auth').hidden = false;
  $('#signup-option').hidden = !state.options.allow_signup;
  $('#forgot-link').hidden = !state.options.password_reset;
  $('#forgot-admin').hidden = Boolean(state.options.password_reset);
  $('#auth-retention').hidden = !state.options.retention_days;
  $('#auth-retention').textContent = retentionText();
  $('input[name="auth_mode"][value="login"]').checked = true;  // e.g. after logging out of a new account
  updateAuthForm();
  showAuthPanel(state.resetToken ? 'reset' : state.needsConsent ? 'consent' : 'login');
}

function showAuthPanel(panel) {
  const panels = { login: '#login-panel', forgot: '#forgot-form', reset: '#reset-form', consent: '#consent-form' };
  for (const [name, selector] of Object.entries(panels)) $(selector).hidden = name !== panel;
  for (const box of $$('#view-auth .error, #view-auth .ok-note')) box.hidden = true;
  const first = $(`${panels[panel]} input:not([type=radio]):not([type=checkbox])`);
  if (first) first.focus();
}

function authMessage(form, message, ok = false) {
  const box = $(ok ? '.ok-note' : '.error', form);
  box.textContent = message;
  box.hidden = false;
}

/* --------------------------------------------------------- privacy (GDPR) */

function consentContent() {
  const days = state.options.retention_days;
  const kept = days ? `that my data sets and results are deleted ${days} days after upload`
    : 'that my data sets and results are kept until I delete them';
  return [
    `I agree that alphaML stores my email address for my account, and ${kept}. I will upload only anonymised `
      + 'data, with no personal or sensitive information such as data that identifies patients, and I am '
      + 'responsible for the data I upload. ',
    el('a', { href: '#/privacy', text: 'Privacy notice' }), ' · ',
    el('a', { href: state.options.privacy.gdpr_url, target: '_blank', rel: 'noopener', text: 'EU GDPR' }),
  ];
}

function fillPrivacy() {
  const privacy = state.options.privacy;
  const days = state.options.retention_days;
  for (const span of $$('.consent-text')) span.replaceChildren(...consentContent());
  for (const link of $$('[data-privacy="gdpr"]')) link.href = privacy.gdpr_url;
  if (privacy.operator) $('[data-privacy="operator"]').textContent = privacy.operator;
  if (privacy.contact) {
    const email = /^[^@\s]+@[^@\s]+$/.test(privacy.contact);
    $('[data-privacy="contact"]').replaceChildren(
      email ? el('a', { href: `mailto:${privacy.contact}`, text: privacy.contact }) : privacy.contact);
  }
  $('[data-privacy="retention"]').textContent = days
    ? `They are deleted automatically ${days} days after upload or submission, or earlier when you delete them `
      + 'or your account.'
    : 'They are kept until you delete them or your account.';
  for (const note of $$('.privacy-upload')) {
    note.textContent = 'Upload only anonymised data: no personal or sensitive information, such as data that '
      + 'identifies patients. You are responsible for the data you upload.';
    note.hidden = !state.options.auth;
  }
}

function showPrivacy() {
  for (const id of ['#view-auth', '#app-main', '#tabs', '#user-menu']) $(id).hidden = true;
  $('#view-privacy').hidden = false;
  window.scrollTo(0, 0);
}

function leavePrivacy() {
  $('#view-privacy').hidden = true;
  if (state.loggedIn) {
    showApp();
    route();
  } else if (state.options.auth) {
    showAuth();
  }
}

async function submitConsent(event) {
  event.preventDefault();
  const form = event.target;
  if (!form.accept_terms.checked) return authMessage(form, 'Please tick the box to agree to the privacy terms.');
  try {
    await postJSON('/api/account/consent', { accept_terms: true });
    form.reset();
    await afterLogin();
  } catch (err) {
    authMessage(form, err.message);
  }
}

/* ---------------------------------------------------------------- log in */

async function submitForgot(event) {
  event.preventDefault();
  const form = event.target;
  $('.error', form).hidden = true;
  try {
    const reply = await postJSON('/api/auth/password-reset', { email: form.email.value.trim() });
    authMessage(form, reply.detail, true);
  } catch (err) {
    authMessage(form, err.message);
  }
}

async function submitReset(event) {
  event.preventDefault();
  const form = event.target;
  if (form.password.value !== form.password2.value) return authMessage(form, 'The passwords do not match.');
  try {
    await postJSON('/api/auth/password-reset/confirm', { token: state.resetToken, password: form.password.value });
    state.resetToken = null;
    form.reset();
    await afterLogin();
    toast('Your new password is set');
  } catch (err) {
    authMessage(form, err.message);
  }
}

function takeResetToken() {
  // Opened from the emailed link: keep the token, and take it out of the address bar and browser history
  const match = location.hash.match(/^#\/reset\/([\w-]+)$/);
  if (!match) return false;
  state.resetToken = match[1];
  history.replaceState(null, '', `${location.pathname}${location.search}#/train`);
  return true;
}

async function afterLogin() {
  // Accounts must agree to the current privacy terms before they can use data and jobs
  const me = await api('/api/me');
  state.needsConsent = !me.consent;
  if (state.needsConsent) return showAuth();
  await enterApp(me.email);
}

function showApp() {
  $('#view-auth').hidden = true;
  $('#app-main').hidden = false;
  $('#tabs').hidden = false;
  $('#user-menu').hidden = !state.options.auth;
}

async function enterApp(email) {
  state.loggedIn = true;
  showApp();
  $('#user-email').textContent = email || '';
  for (const note of $$('.retention-note')) {
    note.hidden = !state.options.retention_days;
    note.textContent = retentionText();
  }
  try {
    await Promise.all([loadDatasets(), loadJobs()]);
    if (location.hash === '#/privacy') showPrivacy(); else route();
  } catch (err) {
    report(err);
  }
}

async function submitAuth(event) {
  event.preventDefault();
  const form = $('#auth-form');
  const errorBox = $('#auth-error');
  const register = authMode() === 'register';
  const fail = message => { errorBox.textContent = message; errorBox.hidden = false; };
  const body = { email: form.email.value.trim(), password: form.password.value };
  if (register) {
    if (form.password.value !== form.password2.value) return fail('The passwords do not match.');
    if (!form.accept_terms.checked) return fail('Please tick the box to agree to the privacy terms.');
    body.signup_code = form.signup_code.value.trim() || null;
    body.accept_terms = true;
  }
  $('#auth-submit').disabled = true;
  try {
    await postJSON(register ? '/api/auth/register' : '/api/auth/login', body);
    form.reset();
    updateAuthForm();
    await afterLogin();
  } catch (err) {
    fail(err.message);
  } finally {
    $('#auth-submit').disabled = false;
  }
}

async function dialogSubmit(dialog, request, onDone) {
  const errorBox = $('.error', dialog);
  errorBox.hidden = true;
  try {
    await request();
    dialog.close();
    $('form', dialog).reset();
    onDone();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  }
}

function initAccount() {
  for (const radio of $$('input[name="auth_mode"]')) radio.addEventListener('change', updateAuthForm);
  $('#auth-form').addEventListener('submit', submitAuth);
  $('#forgot-link').addEventListener('click', () => {
    $('#forgot-form').email.value = $('#auth-form').email.value;
    showAuthPanel('forgot');
  });
  for (const back of $$('.back-to-login')) {
    back.addEventListener('click', () => { state.resetToken = null; showAuthPanel('login'); });
  }
  $('#forgot-form').addEventListener('submit', submitForgot);
  $('#reset-form').addEventListener('submit', submitReset);
  $('#consent-form').addEventListener('submit', submitConsent);
  $('#consent-logout').addEventListener('click', async () => {
    try { await postJSON('/api/auth/logout', {}); } catch (_) { /* logged out anyway */ }
    state.needsConsent = false;
    showAuth();
  });
  $('#privacy-back').addEventListener('click', event => {
    if (history.length > 1) {
      event.preventDefault();
      history.back();
    }
  });
  for (const button of $$('dialog button[value="cancel"]')) {
    button.addEventListener('click', () => button.closest('dialog').close());
  }
  const closeMenu = () => { $('#user-menu').open = false; };
  $('#menu-logout').addEventListener('click', async () => {
    closeMenu();
    try { await postJSON('/api/auth/logout', {}); } catch (_) { /* logged out anyway */ }
    state.needsConsent = false;
    showAuth();
  });
  $('#user-menu a[href="#/privacy"]').addEventListener('click', closeMenu);
  $('#menu-password').addEventListener('click', () => { closeMenu(); $('#password-dialog').showModal(); });
  $('#menu-delete').addEventListener('click', () => { closeMenu(); $('#delete-dialog').showModal(); });
  $('#password-form').addEventListener('submit', event => {
    event.preventDefault();
    const form = event.target;
    dialogSubmit($('#password-dialog'), () => postJSON('/api/account/password', {
      current_password: form.current_password.value, new_password: form.new_password.value,
    }), () => toast('Password changed'));
  });
  $('#delete-form').addEventListener('submit', event => {
    event.preventDefault();
    dialogSubmit($('#delete-dialog'), () => postJSON('/api/account/delete', { password: event.target.password.value }),
      () => { showAuth(); toast('Your account and data were deleted'); });
  });
}

/* ------------------------------------------------------------------ start */

async function start() {
  try {
    state.options = await api('/api/options');
  } catch (err) {
    document.body.replaceChildren(el('p', { class: 'error', style: 'padding:2rem', text: `Cannot reach the alphaML server: ${err.message}` }));
    return;
  }
  initTrainForm();
  initPredictForm();
  initAccount();
  fillPrivacy();
  window.addEventListener('hashchange', () => {
    if (location.hash === '#/privacy') showPrivacy();
    else if (state.options.auth && takeResetToken()) showAuth();
    else if (!$('#view-privacy').hidden) leavePrivacy();
    else if (state.loggedIn) route();
  });
  if (state.options.auth && takeResetToken()) {
    showAuth();
    return;
  }
  // Keep the job list and the nav counter fresh while jobs are running
  setInterval(() => {
    if (state.loggedIn && (state.jobs.some(isActive) || state.view === 'jobs')) loadJobs();
  }, 5000);
  if (!state.options.auth) return enterApp(null);
  try {
    await afterLogin();
  } catch (err) {
    if (!err.silent) report(err);  // otherwise the login screen is shown
  }
  if (location.hash === '#/privacy') showPrivacy();
}

start();
