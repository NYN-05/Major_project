import React, { useEffect, useState } from "react";

const NUM = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");

function LineChart({ data, series }) {
  if (!data.length) return <p className="axis">No training log found — run the pipeline first.</p>;
  const w = 640, h = 220, pad = 34;
  const xs = data.map((d) => d.epoch);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const all = series.flatMap((s) => data.map((d) => d[s.key]));
  const minV = Math.min(...all), maxV = Math.max(...all);
  const px = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (w - 2 * pad);
  const py = (v) => h - pad - ((v - minV) / (maxV - minV || 1)) * (h - 2 * pad);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label={series.map((s) => s.label).join(" and ")}>
      {[0.25, 0.5, 0.75, 1].map((t) => (
        <line key={t} x1={pad} x2={w - pad} y1={pad + t * (h - 2 * pad)} y2={pad + t * (h - 2 * pad)} className="gridln" />
      ))}
      {series.map((s, i) => (
        <polyline key={s.key} className={`curve c${i}`} points={data.map((d) => `${px(d.epoch)},${py(d[s.key])}`).join(" ")} />
      ))}
      <text x={pad} y={14} className="axis">{series.map((s) => s.label).join(" / ")}</text>
      <text x={w - pad} y={h - 6} className="axis" textAnchor="end">EPOCH</text>
    </svg>
  );
}

function Bars({ items }) {
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <div className="bars">
      {items.map((i, idx) => (
        <div key={idx} className="brow">
          <span className="bname" title={i.label}>{i.label}</span>
          <div className="btrack"><div className="bfill" style={{ width: `${(i.value / max) * 100}%` }} /></div>
          <span className="bval">{i.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [log, setLog] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/metrics").then((r) => r.json()),
      fetch("/api/training_log").then((r) => r.json()),
    ])
      .then(([m, l]) => { setMetrics(m); setLog(l); })
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="wrap error-box">
        <h1>Link down — API unreachable</h1>
        <p>{error}</p>
        <p>Start the server with <code>python quantum/app.py</code>, then reload this page.</p>
      </div>
    );
  }

  const q = metrics?.quantum || {};
  const hasQ = Object.keys(q).length > 1;
  const cm = q.confusion_matrix || {};
  const dec = q.decisions || {};
  const sel = metrics?.selection || {};
  const features = sel.selected_features || [];
  const marginals = Array.isArray(sel.marginal_probabilities) ? sel.marginal_probabilities : [];
  const featureNames = Array.isArray(sel.feature_names) ? sel.feature_names : [];
  const baselines = metrics?.baselines || {};
  const hasBase = Object.keys(baselines).length > 0;

  const acc = hasQ ? NUM(q.accuracy, 2) : "—";

  return (
    <>
      <div className="topbar">
        <div className="wrap">
          <b>QML//DEMO</b>
          <span>PENNYLANE · PYTORCH · LIGHTNING</span>
        </div>
      </div>

      <section className="hero">
        <div className="wrap">
          <div className="eyebrow">HYBRID QUANTUM-CLASSICAL INFERENCE</div>
          <h1>
            PULSE <span className="lit">FORENSICS</span><br />
            <span className="ghost">DEEPFAKE DETECTION VIA rPPG</span>
          </h1>
          <p className="sub">
            A QAOA-selected subset of 9 pulse-signal features feeds a hybrid variational
            quantum classifier that separates real video from synthetic manipulation.
          </p>

          <div className="meter">
            <div className="scanline" aria-hidden="true" />
            <div className="meter-grid">
              <div className="meter-main">
                <div className="tag">TEST ACCURACY — HYBRID VQC</div>
                <div className="meter-num">{acc}<span className="unit"> / 1.00</span></div>
                <div className="cap">{hasQ ? q.model : "No trained model found — run the pipeline."}</div>
              </div>
              <div className="meter-cell">
                <span className="k">F1 SCORE</span>
                <span className="v">{hasQ ? NUM(q.f1, 2) : "—"}</span>
              </div>
              <div className="meter-cell">
                <span className="k">AUC-ROC</span>
                <span className="v">{hasQ ? NUM(q.auc_roc, 2) : "—"}</span>
              </div>
              <div className="meter-cell">
                <span className="k">CALIBRATION ECE</span>
                <span className="v">{hasQ ? NUM(q.ece, 2) : "—"}</span>
              </div>
            </div>
          </div>

          <div className="stages">
            {[
              ["01", "SELECT", "QAOA picks 6 of 9 features"],
              ["02", "TRAIN", "Hybrid VQC · focal loss · 40 epochs"],
              ["03", "VERIFY", "Metrics, calibration, decision bins"],
              ["04", "COMPARE", "MLP / RandomForest / SVM baselines"],
            ].map(([no, nm, ds]) => (
              <div className="stage" key={no}>
                <span className="no">{no}</span>
                <span className="nm">{nm}</span>
                <span className="ds">{ds}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="wrap">
        <section className="block" id="results">
          <div className="sec-head"><span className="idx">01</span><h2>Result deck</h2><span className="kick">TEST SPLIT</span></div>
          <div className="panel">
            {hasQ ? (
              <>
                <div className="cards">
                  <div className="mcard accent"><div className="lbl">ACCURACY</div><div className="val">{NUM(q.accuracy)}</div><div className="sub">n = 160 test samples</div></div>
                  <div className="mcard"><div className="lbl">F1 SCORE</div><div className="val">{NUM(q.f1)}</div><div className="sub">binary, real vs fake</div></div>
                  <div className="mcard"><div className="lbl">PRECISION</div><div className="val">{NUM(q.precision)}</div><div className="sub">positives that are real</div></div>
                  <div className="mcard"><div className="lbl">RECALL</div><div className="val">{NUM(q.recall)}</div><div className="sub">reals that are caught</div></div>
                  <div className="mcard accent"><div className="lbl">AUC-ROC</div><div className="val">{NUM(q.auc_roc)}</div><div className="sub">threshold-free ranking</div></div>
                  <div className="mcard"><div className="lbl">ECE</div><div className="val">{NUM(q.ece)}</div><div className="sub">expected calibration error</div></div>
                </div>

                <div className="decisions">
                  <span className="chip real"><span className="n">{dec.real_count ?? "?"}</span>DECLARED REAL</span>
                  <span className="chip unc"><span className="n">{dec.uncertain_count ?? "?"}</span>UNCERTAIN</span>
                  <span className="chip fake"><span className="n">{dec.fake_count ?? "?"}</span>DECLARED FAKE</span>
                  <span className="chip">THRESHOLDS — FAKE ≤ {q.thresholds?.fake_max_prob} / REAL ≥ {q.thresholds?.real_min_prob}</span>
                </div>

                <div className="cm">
                  <div className="cm-cell"><div className="cm-val">{cm.tn ?? "?"}</div><div className="cm-lbl">TRUE NEGATIVE · FAKE OK</div></div>
                  <div className="cm-cell hot"><div className="cm-val">{cm.fp ?? "?"}</div><div className="cm-lbl">FALSE POSITIVE</div></div>
                  <div className="cm-cell hot"><div className="cm-val">{cm.fn ?? "?"}</div><div className="cm-lbl">FALSE NEGATIVE</div></div>
                  <div className="cm-cell"><div className="cm-val">{cm.tp ?? "?"}</div><div className="cm-lbl">TRUE POSITIVE · REAL OK</div></div>
                </div>
              </>
            ) : (
              <p className="axis">No quantum metrics on disk. Run <code>python quantum/run_quantum.py --all</code> and refresh.</p>
            )}
          </div>
        </section>

        <section className="block" id="charts">
          <div className="sec-head"><span className="idx">02</span><h2>Charts</h2><span className="kick">OUTPUT /</span></div>
          <div className="panel">
            <div className="plots">
              <figure className="plot">
                <img src="/api/plots/roc" alt="ROC curve of the hybrid VQC" />
                <figcaption>ROC — HYBRID VQC</figcaption>
              </figure>
              <figure className="plot">
                <img src="/api/plots/confusion" alt="Confusion matrix of the hybrid VQC" />
                <figcaption>CONFUSION — HYBRID VQC</figcaption>
              </figure>
            </div>
            <div className="curve-block">
              <h3>Training curves — loss</h3>
              <LineChart data={log} series={[{ key: "train_loss", label: "TRAIN LOSS" }, { key: "val_loss", label: "VAL LOSS" }]} />
            </div>
            <div className="curve-block">
              <h3>Training curves — accuracy</h3>
              <LineChart data={log} series={[{ key: "train_acc", label: "TRAIN ACC" }, { key: "val_acc", label: "VAL ACC" }]} />
            </div>
          </div>
        </section>

        <section className="block" id="features">
          <div className="sec-head"><span className="idx">03</span><h2>Feature selection</h2><span className="kick">QAOA · p={sel.p_layers ?? 3}</span></div>
          <div className="panel two">
            <div className="blk">
              <h3>What QAOA selected</h3>
              <p>
                Mutual information scores relevance, correlation scores redundancy; the
                pair folds into a spin Hamiltonian that a p={sel.p_layers ?? 3} QAOA circuit
                minimizes. The surviving features:
              </p>
              <div className="sel">
                {features.map((f) => <span key={f} className="f">{f}</span>)}
              </div>
              <p style={{ marginTop: 14 }}>
                Target: {sel.target_features ?? "—"} of 9 features · energy: {sel.expected_cost_energy != null ? NUM(sel.expected_cost_energy, 2) : "—"}
              </p>
            </div>
            <div className="blk">
              <h3>Selection marginals</h3>
              <p>Probability each feature lands in the optimized subset.</p>
              {marginals.length ? (
                <Bars items={marginals.map((v, i) => ({ label: (featureNames[i] || i).replace(/_/g, " "), value: v }))} />
              ) : <p className="axis">No selection data.</p>}
            </div>
          </div>
        </section>

        {hasBase && (
          <section className="block" id="baselines">
            <div className="sec-head"><span className="idx">04</span><h2>Classical baselines</h2><span className="kick">FULL 9-FEATURE SET</span></div>
            <div className="panel">
              <table className="tbl">
                <thead>
                  <tr><th>Model</th><th>Accuracy</th><th>F1</th><th>AUC-ROC</th><th>Precision</th><th>Recall</th></tr>
                </thead>
                <tbody>
                  {Object.entries(baselines).map(([name, m]) => (
                    <tr key={name}>
                      <td>{m.model || name}</td>
                      <td>{NUM(m.accuracy)}</td>
                      <td>{NUM(m.f1)}</td>
                      <td>{NUM(m.auc_roc)}</td>
                      <td>{NUM(m.precision)}</td>
                      <td>{NUM(m.recall)}</td>
                    </tr>
                  ))}
                  {hasQ && (
                    <tr>
                      <td><b style={{ color: "var(--ember)" }}>{q.model}</b></td>
                      <td><b style={{ color: "var(--ember)" }}>{NUM(q.accuracy)}</b></td>
                      <td><b style={{ color: "var(--ember)" }}>{NUM(q.f1)}</b></td>
                      <td><b style={{ color: "var(--ember)" }}>{NUM(q.auc_roc)}</b></td>
                      <td><b style={{ color: "var(--ember)" }}>{NUM(q.precision)}</b></td>
                      <td><b style={{ color: "var(--ember)" }}>{NUM(q.recall)}</b></td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="block" id="impl">
          <div className="sec-head"><span className="idx">05</span><h2>Implementation</h2><span className="kick">PIPELINE NOTES</span></div>
          <div className="panel two">
            <div className="blk">
              <h3>QAOA — the selector</h3>
              <p>
                Feature subsetting becomes a QUBO: maximize mutual information with the
                label, minimize pairwise correlation, and hold the subset at size k. COBYLA
                tunes the circuit angles; the most probable bitstring becomes the subset.
              </p>
            </div>
            <div className="blk">
              <h3>VQC — the classifier</h3>
              <p>
                Selected features are angle-embedded into 6 qubits, run through strongly
                entangling layers, and read out as Pauli-Z expectations into a small
                classical head — trained with balanced focal loss and label smoothing.
              </p>
            </div>
          </div>
        </section>

        <div className="stripe" aria-hidden="true" />

        <footer>
          Demo frontend · data served from <code>quantum/output</code> by Flask · retrain with <code>python quantum/run_quantum.py --all</code>
        </footer>
      </div>
    </>
  );
}
