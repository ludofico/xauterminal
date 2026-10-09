"use client";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { createChart, CandlestickSeries, LineSeries, type IChartApi } from "lightweight-charts";

type Any = any;
const fmt = (n: number | null | undefined, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
const cls = (v: number) => (v > 0 ? "up" : v < 0 ? "down" : "mute");

const Chart = memo(function Chart({ snap }: { snap: Any }) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<Any>({});
  useEffect(() => {
    if (!el.current) return;
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const c = createChart(el.current, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: dark ? "#7d8793" : "#66707a" },
      grid: { vertLines: { visible: false }, horzLines: { color: dark ? "#1f262e" : "#eceff1" } },
      timeScale: { timeVisible: true },
    });
    series.current.candles = c.addSeries(CandlestickSeries, { upColor: "#2bb673", downColor: "#e5534b", wickUpColor: "#2bb673", wickDownColor: "#e5534b", borderVisible: false });
    series.current.med = c.addSeries(LineSeries, { color: "#d4a94a", lineWidth: 2 });
    series.current.q10 = c.addSeries(LineSeries, { color: "#d4a94a55", lineWidth: 1 });
    series.current.q90 = c.addSeries(LineSeries, { color: "#d4a94a55", lineWidth: 1 });
    chart.current = c;
    return () => c.remove();
  }, []);
  const loaded = useRef(false);
  useEffect(() => {
    // Perf: one full load, then only the last 2 bars with series.update (no 400-bar redraw every tick).
    const tick = () => fetch(`/api/candles?n=${loaded.current ? 2 : 1000}`).then(r => r.json()).then(rows => {
      if (!rows.length) return;
      if (loaded.current) rows.forEach((b: Any) => series.current.candles.update(b));
      else { series.current.candles.setData(rows); loaded.current = true; }
    }).catch(() => {});
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    // Overlays change rarely (levels, forecast): redraw only when they change.
    const load = () => fetch("/api/candles?n=1").then(r => r.json()).then(rows => {
      if (!rows.length) return;
      const s = series.current.candles;
      (s._lines || []).forEach((l: Any) => s.removePriceLine(l));
      s._lines = Object.entries(snap?.levels || {}).map(([name, price]) =>
        s.createPriceLine({ price, color: String(name).includes("high") ? "#e5534b88" : "#2bb67388", lineWidth: 1, lineStyle: 2, title: name }));
      const fc = snap?.forecast;
      if (fc) {
        const t0 = rows[rows.length - 1].time, step = 900;
        // anchor the forecast on the live price: removes futures/spot basis and staleness between 15-min refreshes
        const shift = fc.base ? rows[rows.length - 1].close - fc.base : 0;
        const mk = (arr: number[]) => arr.map((v, i) => ({ time: t0 + (i + 1) * step, value: v + shift }));
        series.current.med.setData(mk(fc.median)); series.current.q10.setData(mk(fc.q10)); series.current.q90.setData(mk(fc.q90));
      }
    }).catch(() => {});
    load();
  }, [JSON.stringify(snap?.levels), snap?.forecast?.median?.[0], Math.round(snap?.price || 0)]);
  return <div ref={el} style={{ height: 380 }} />;
}, (a, b) => JSON.stringify(a.snap?.levels) === JSON.stringify(b.snap?.levels) && a.snap?.forecast?.median?.[0] === b.snap?.forecast?.median?.[0]);

function Bar({ v }: { v: number }) {
  const w = Math.min(50, Math.abs(v) * 50);
  return <div className="bar"><i style={{ left: v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? "var(--up)" : "var(--down)" }} /></div>;
}

const Graph = memo(function Graph({ g }: { g: Any }) {
  if (!g?.nodes?.length) return <div className="mute">in calcolo…</div>;
  const R = 90, cx = 130, cy = 110;
  const pos = Object.fromEntries(g.nodes.map((n: string, i: number) => [n, [cx + R * Math.cos((2 * Math.PI * i) / g.nodes.length), cy + R * Math.sin((2 * Math.PI * i) / g.nodes.length)]]));
  return (
    <svg viewBox="0 0 260 220" style={{ width: "100%", height: 220 }} role="img" aria-label="Grafo di dipendenza tra coppie XAU">
      {g.edges.map((e: Any, i: number) => <line key={i} x1={pos[e.a][0]} y1={pos[e.a][1]} x2={pos[e.b][0]} y2={pos[e.b][1]}
        stroke={e.w > 0 ? "var(--up)" : "var(--down)"} strokeWidth={1 + 6 * Math.abs(e.w)} opacity={0.7} />)}
      {g.nodes.map((n: string) => <g key={n}><circle cx={pos[n][0]} cy={pos[n][1]} r={16} fill="var(--card)" stroke="var(--gold)" />
        <text x={pos[n][0]} y={pos[n][1] + 3} fontSize={8} textAnchor="middle" fill="var(--text)">{n.replace("XAU", "")}</text></g>)}
    </svg>
  );
}, (a, b) => JSON.stringify(a.g) === JSON.stringify(b.g));

export default function Page() {
  const [s, setS] = useState<Any>({});
  const [news, setNews] = useState<Any[]>([]);
  const [q, setQ] = useState("");
  const [ask, setAsk] = useState(""); const [ans, setAns] = useState<Any>(null); const [busy, setBusy] = useState(false);

  useEffect(() => {
    const loadNews = () => fetch("/api/news?limit=300").then(r => r.json()).then(setNews).catch(() => {});
    loadNews();
    const newsId = setInterval(loadNews, 10000);  // picks up classification that lands after a headline first appears
    const es = new EventSource("/api/stream");
    es.onmessage = e => {
      let d: Any;
      try { d = JSON.parse(e.data); } catch { return; }  // never crash the page on one bad frame
      setS(d);
      if (d.news_delta?.length) setNews(n => [...d.news_delta.reverse(), ...n].slice(0, 500));
    };
    return () => { es.close(); clearInterval(newsId); };
  }, []);

  const c = s.confluence || {}; const sys = s.system || {}; const fc = s.forecast; const cross = s.cross || {};
  const shown = useMemo(() => (q ? news.filter(n => n.title.toLowerCase().includes(q.toLowerCase())) : news).slice(0, 150), [news, q]);
  const themes = useMemo(() => {
    const t: Record<string, number> = {};
    news.slice(0, 80).forEach(n => { if (n.nlp?.bias) t[n.nlp.theme] = (t[n.nlp.theme] || 0) + ((n.nlp.bias.bull || 0) - (n.nlp.bias.bear || 0)); });
    return t;
  }, [news]);

  return (
    <main>
      <div className="ticker mono">
        <b className="gold">XAU TERMINAL</b>
        {Object.entries(s.ticker || {}).map(([k, v]) => <span key={k}>{k} <b>{fmt(v as number, k === "XAUJPY" ? 0 : 2)}</b> <span className={cls(cross.returns?.[k] || 0)}>{cross.returns?.[k] != null ? `${(100 * cross.returns[k]).toFixed(2)}%` : ""}</span></span>)}
        <span className="mute">{s.source} · NY {s.session?.ny_time}</span>
      </div>
      <div className="grid">
        <section className="card c8 r3"><h3>XAUUSD M1 · livelli di liquidità · forecast TimesFM</h3><Chart snap={s} /></section>

        <section className="card c4 r2">
          <h3>Confluenza</h3>
          <div className="big mono">{c.score ?? "—"}<span className="mute" style={{ fontSize: 14 }}>/100</span>
            <span className={cls(c.direction)} style={{ fontSize: 18, marginLeft: 10 }}>{c.direction > 0 ? "LONG" : c.direction < 0 ? "SHORT" : "FLAT"}</span></div>
          {c.no_trade && <div className="pill notrade" style={{ marginTop: 8 }}>NO TRADE · {c.no_trade_reasons?.join(" · ")}</div>}
          {s.gate && <div className="mute" style={{ marginTop: 6 }}>Gate System One ({s.gate.engine}): P={fmt(s.gate.p_take)} dir {s.gate.direction > 0 ? "long" : "short"}</div>}
          <table style={{ marginTop: 8 }}><tbody>
            {(c.components || []).map((k: Any) => <tr key={k.name}><td style={{ width: 90 }}>{k.name}</td><td style={{ width: 90 }}><Bar v={k.value} /></td><td className="mute" style={{ paddingLeft: 8 }}>{k.note}</td></tr>)}
          </tbody></table>
          <div style={{ marginTop: 8 }} className="mono">
            {c.trapped && <div>Intrappolati: {c.trapped}</div>}
            <div>Invalidazione {fmt(c.invalidation)} · Target liquidità {fmt(c.target_liquidity)}</div>
            {s.mc?.p_target_first != null && <div>P(target prima dello stop) <b>{(100 * s.mc.p_target_first).toFixed(0)}%</b></div>}
          </div>
        </section>

        <section className="card c4">
          <h3>Multi-timeframe · macro → micro</h3>
          {s.mtf ? <>
            <div>Bias HTF <b className={cls(s.mtf.htf_bias)}>{fmt(s.mtf.htf_bias)}</b> · trigger LTF <b className={cls(s.mtf.ltf_bias)}>{fmt(s.mtf.ltf_bias)}</b>
              {s.mtf.aligned ? <span className="pill up" style={{ marginLeft: 6 }}>allineati</span> : <span className="pill notrade" style={{ marginLeft: 6 }}>non allineati</span>}</div>
            <table style={{ marginTop: 6 }}><tbody>
              <tr className="mute"><td>TF</td><td>trend</td><td>struttura</td><td>BOS</td><td>zona</td><td>RSI</td></tr>
              {["D1", "H4", "H1", "M15", "M5", "M1"].map(tf => { const r = s.mtf.tf?.[tf]; if (!r) return null;
                const arrow = (v: number) => <span className={cls(v)}>{v > 0 ? "▲" : v < 0 ? "▼" : "—"}</span>;
                return <tr key={tf} className="mono"><td>{tf}</td><td>{arrow(r.trend)}</td><td>{arrow(r.structure)}</td><td>{arrow(r.bos)}</td>
                  <td className={r.zone === "discount" ? "up" : "down"}>{r.zone} {(100 * r.pd_pct).toFixed(0)}%</td><td>{r.rsi ?? "—"}</td></tr>; })}
            </tbody></table>
          </> : <div className="mute">in calcolo…</div>}
        </section>

        <section className="card c4">
          <h3>Sistema</h3>
          <div>Regime <b>{sys.regime?.regime || "—"}</b></div>
          {Object.entries(sys.regime?.probs || {}).map(([k, v]) => <div key={k} className="mute mono">{k} {(100 * (v as number)).toFixed(0)}%</div>)}
          {sys.fair_value && <div style={{ marginTop: 6 }}>Fair value Kalman <b className="mono">{fmt(sys.fair_value.fair_value)}</b> · z <span className={cls(-sys.fair_value.z)}>{sys.fair_value.z}</span></div>}
          {fc && <div>TimesFM 8h <span className={cls(fc.move_pct)}>{fc.move_pct}%</span> · banda {fc.band_pct}% · hit-rate {fc.hit_rate ?? "n/d"}</div>}
        </section>

        <section className="card c4">
          <h3>Forza oro vs valute</h3>
          <div>Breadth <b>{cross.breadth != null ? `${(100 * cross.breadth).toFixed(0)}%` : "—"}</b> · USDX <span className={cls(-(cross.usdx_ret || 0))}>{fmt(100 * (cross.usdx_ret || 0))}%</span></div>
          {cross.usd_driven && <div className="pill notrade">solo debolezza USD</div>}
          {Object.entries(cross.z || {}).map(([k, v]) => <div key={k} style={{ display: "grid", gridTemplateColumns: "60px 1fr 40px", gap: 8, alignItems: "center" }}><span>{k.replace("XAU", "")}</span><Bar v={Math.max(-1, Math.min(1, (v as number) / 2))} /><span className="mono">{fmt(v as number)}</span></div>)}
          {Object.keys(s.smt?.divergent || {}).length > 0 && <div style={{ marginTop: 6 }}>SMT: {Object.entries(s.smt.divergent).map(([k, v]) => `${k} → ${(v as string[]).join(",")}`).join(" · ")}</div>}
        </section>

        <section className="card c4"><h3>Grafo di dipendenza (graphical lasso)</h3><Graph g={sys.graph} /></section>

        <section className="card c4">
          <h3>Grafo causale (PCMCI+, giornaliero)</h3>
          {(sys.causal || []).slice(0, 10).map((l: Any, i: number) => <div key={i} className="mono">{l.src} → {l.dst} <span className="mute">lag {l.lag}</span> <span className={cls(l.val)}>{l.val}</span></div>)}
          {!sys.causal?.length && <div className="mute">in calcolo (refresh orario)…</div>}
        </section>

        <section className="card c4">
          <h3>Posizionamento · Cina</h3>
          {s.cot && <div>COT net {s.cot.net?.toLocaleString()} ({s.cot.wow > 0 ? "+" : ""}{s.cot.wow?.toLocaleString()}) · index 52w {fmt(s.cot.cot_index_52w)}</div>}
          {s.oanda ? <div>Retail long {(100 * s.oanda.retail_long_pct).toFixed(0)}% · pool buy {s.oanda.buy_side_pools?.join(", ")} · sell {s.oanda.sell_side_pools?.join(", ")}</div> : null}
          {s.sge && <div>Premio SGE <b className={cls(s.sge.premium_pct)}>{s.sge.premium_pct}%</b></div>}
        </section>

        <section className="card c4">
          <h3>Macro sentiment (ultime 80 news)</h3>
          {Object.entries(themes).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => <div key={k} style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: 8, alignItems: "center" }}><span>{k}</span><Bar v={Math.max(-1, Math.min(1, v / 5))} /></div>)}
        </section>

        <section className="card c7 r3">
          <h3>News realtime · {news.length}</h3>
          <input placeholder="Cerca…" value={q} onChange={e => setQ(e.target.value)} />
          <ul className="news">
            {shown.map(n => {
              const b = n.nlp?.bias || {}; const net = (b.bull || 0) - (b.bear || 0);
              return <li key={n.id}>
                <span className="mute mono">{n.ts?.slice(11, 16)}Z · {n.source}</span>{" "}
                {n.nlp && <span className={`pill ${cls(net)}`}>{net > 0 ? "bull" : net < 0 ? "bear" : "neutro"} {fmt(Math.max(b.bull || 0, b.bear || 0))}{n.nlp.calibrated ? "" : "*"}</span>}
                {n.nlp && <span className="pill">{n.nlp.theme}</span>}
                <div><a href={n.url} target="_blank" rel="noreferrer">{n.title}</a></div>
              </li>;
            })}
          </ul>
          <div className="mute">* non calibrato (fallback). Fonti: {Object.entries(s.news_status || {}).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
        </section>

        <section className="card c5">
          <h3>Chiedi alle news</h3>
          <form onSubmit={e => { e.preventDefault(); setBusy(true); fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ q: ask }) }).then(r => r.json()).then(setAns).finally(() => setBusy(false)); }} style={{ display: "flex", gap: 6 }}>
            <input value={ask} onChange={e => setAsk(e.target.value)} placeholder="Perché l'oro sale oggi?" /><button disabled={busy}>{busy ? "…" : "Chiedi"}</button>
          </form>
          {ans && <p style={{ whiteSpace: "pre-wrap" }}>{ans.answer}</p>}
        </section>

        <section className="card c5">
          <h3>Calendario macro</h3>
          {(s.calendar || []).slice(0, 8).map((e: Any, i: number) => <div key={i} className="mono"><span className={e.impact === "High" ? "down" : "mute"}>●</span> {e.ts?.slice(5, 16).replace("T", " ")}Z {e.country} {e.title}</div>)}
        </section>

        <section className="card c5">
          <h3>Stato sorgenti</h3>
          <div>Feed prezzi <b>{s.source}</b>{s.stream ? <span className="mute"> · {s.stream.series} serie · {s.stream.ticks} tick</span> : null}</div>
          <div>Classificatore <b>{s.news_status?.laya === "ready" ? "Laya (calibrato)" : s.news_status?.jev?.startsWith("ok") ? "Jev (calibrato)" : "keyword (non calibrato)"}</b>
            <span className="mute"> · laya: {s.news_status?.laya} · jev: {s.news_status?.jev}</span></div>
          {Object.entries(s.errors || {}).map(([k, v]) => <div key={k} className="mute">{k}: {String(v)}</div>)}
          {!Object.keys(s.errors || {}).length && <div className="up">tutte ok</div>}
        </section>
      </div>
    </main>
  );
}
