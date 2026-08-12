const fileUrl = (rel) => `/api/files?rel=${encodeURIComponent(rel)}`;

async function detect(videoFile) {
  const res = await fetch("/api/detect", {
    method: "POST",
    headers: { "X-Filename": videoFile.name },
    body: videoFile,
  });
  if (res.status === 413 || res.status === 415 || res.status === 429) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `upload rejected (${res.status})`);
  }
  if (!res.ok) throw new Error(`upload failed (${res.status})`);
  return res.json();
}

function stream(jobId, handlers) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    const { kind, data } = msg;
    if (handlers[kind]) handlers[kind](data);
    if (kind === "result" || kind === "error") es.close();
  };
  es.onerror = () => {
    if (handlers.error) handlers.error("progress stream lost");
    es.close();
  };
  return es;
}

async function previous() {
  const res = await fetch("/api/previous");
  if (!res.ok) return { result: null };
  return res.json();
}

async function health() {
  const res = await fetch("/api/health");
  if (!res.ok) return null;
  return res.json();
}

async function artifacts(dir) {
  const res = await fetch(`/api/artifacts?dir=${encodeURIComponent(dir)}`);
  if (!res.ok) return { files: [] };
  return res.json();
}

export { artifacts, detect, fileUrl, health, previous, stream };