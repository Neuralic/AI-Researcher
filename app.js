// app.js
const form = document.getElementById('queryForm');
const loader = document.getElementById('loader');
const result = document.getElementById('result');

form.addEventListener('submit', async e => {
  e.preventDefault();
  loader.classList.remove('hidden');
  result.classList.add('hidden');

  const payload = {
    query: document.getElementById('query').value.trim(),
    mode_override: document.getElementById('mode').value || undefined
  };

  try {
    const res = await fetch('/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    result.textContent = data.content;
  } catch (err) {
    result.textContent = 'Error: ' + err.message;
  } finally {
    loader.classList.add('hidden');
    result.classList.remove('hidden');
  }
});