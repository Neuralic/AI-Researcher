// app.js
const API_URL = '/research';

const form    = document.getElementById('queryForm');
const loader  = document.getElementById('loader');
const result  = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  loader.classList.remove('hidden');
  result.classList.add('hidden');

  const payload = {
    query: document.getElementById('query').value.trim(),
    mode_override: document.getElementById('mode').value || undefined
  };

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || res.statusText);
    }

    const data = await res.json();
    result.innerHTML = data.content
      .replace(/\n/g, '<br>')
      .replace(/!\!CHAOS\!\!/g, '<span style="color:#ef4444;font-weight:bold">!!CHAOS!!</span>');
  } catch (err) {
    result.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
  } finally {
    loader.classList.add('hidden');
    result.classList.remove('hidden');
  }
});