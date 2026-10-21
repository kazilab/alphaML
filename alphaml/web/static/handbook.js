/* Public settings only; the handbook never requests account, data set or job records. */
(() => {
  'use strict';
  const printButton = document.getElementById('print-handbook');
  printButton.hidden = false;
  printButton.addEventListener('click', () => window.print());
  const setText = (id, value) => { document.getElementById(id).textContent = value; };
  fetch('/api/options', { credentials: 'omit' })
    .then(response => {
      if (!response.ok) throw new Error('Settings unavailable');
      return response.json();
    })
    .then(options => {
      const mb = bytes => Number.isFinite(bytes) ? `${Number((bytes / 1048576).toFixed(2))} MB` : null;
      if (mb(options.max_upload_bytes)) setText('max-upload', mb(options.max_upload_bytes));
      if (mb(options.upload_chunk_bytes)) setText('upload-chunk', mb(options.upload_chunk_bytes));
      if (Number.isInteger(options.max_running_jobs)) setText('max-running', options.max_running_jobs);
      if (options.auth === false) {
        setText('account-mode', 'Accounts are disabled on this server. Anyone who can open the application can see its data sets, jobs and results. Use a shared server only according to its operator’s access and data policy.');
      }
      if (Number.isFinite(options.retention_days)) {
        setText('retention-text', options.retention_days > 0
          ? `this server removes data sets and jobs ${options.retention_days} days after upload/submission. Cleanup runs periodically and protects active inputs while in use.`
          : 'automatic age-based deletion of data sets and jobs is disabled on this server. Delete items manually when no longer needed.');
      }
      setText('model-upload-note', options.allow_model_upload
        ? 'This server permits model-bundle uploads. Use only trusted alphaML 2.0 *_bundle.pkl files: loading a pickle can execute code. Models from alphaML 0.x require retraining. Choosing your own finished training job is the usual workflow.'
        : 'Model-bundle uploads are disabled on this server. Predict using your own succeeded training job. Download bundles for archival or trusted use in a compatible Python environment; models from alphaML 0.x require retraining.');
      const contact = options.privacy && options.privacy.contact;
      if (typeof contact === 'string' && contact.trim()) setText('support-contact', contact.trim());
      setText('configuration-status', 'Live upload, concurrency, account, retention, model-upload and contact settings loaded from this server. Modeling defaults below describe alphaML 0.2.0.');
    })
    .catch(() => { /* The complete handbook remains readable with the documented defaults. */ });
})();
