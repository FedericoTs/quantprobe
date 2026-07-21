"""quantprobe dashboard — the law, live.

One command: plans the best placement, launches llama-server with those flags, and serves a local
page where you chat with the model while every reply is scored against the law's prediction —
predicted vs measured tok/s, live. Colibri-style dashboard energy, but the star is falsifiability:
the page is a running experiment, not a status screen.
"""
from __future__ import annotations
import json, os, subprocess, sys, threading, time, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import runtime

DASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>quantprobe — the law, live</title>
<style>
:root{--g:#fafaf7;--p:#fff;--ink:#16181d;--sub:#5c6066;--line:#e5e4df;--acc:#0f766e;
--accsoft:#e6f2f0;--warn:#b45309;--mono:Consolas,ui-monospace,monospace}
@media(prefers-color-scheme:dark){:root{--g:#14161a;--p:#1c1f24;--ink:#e8e7e2;--sub:#9aa0a8;
--line:#2e3238;--acc:#35b8a6;--accsoft:#12332f;--warn:#e0a458}}
body{margin:0;background:var(--g);color:var(--ink);font-family:"Segoe UI",system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:22px 16px}
h1{font-size:21px;margin:0 0 4px}.sub{color:var(--sub);font-size:13px;margin-bottom:16px}
.law{font-family:var(--mono);font-size:12.5px;color:var(--acc);margin-bottom:14px}
.cols{display:grid;grid-template-columns:340px 1fr;gap:16px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
.panel{background:var(--p);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--sub);font-weight:600;margin-bottom:8px}
.gauge{display:flex;gap:18px;align-items:baseline}
.gauge .n{font-family:var(--mono);font-size:44px;font-weight:700;color:var(--acc);font-variant-numeric:tabular-nums}
.gauge .lab{font-size:11px;color:var(--sub)}
.delta{font-family:var(--mono);font-size:15px;font-weight:600}
.tier{margin:8px 0}.tier .l{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--sub);margin-bottom:3px}
.bar{height:11px;background:var(--g);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--acc)}
.log{font-family:var(--mono);font-size:11.5px;color:var(--sub);max-height:120px;overflow-y:auto;margin-top:8px}
.chat{display:flex;flex-direction:column;height:520px}
.msgs{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-bottom:8px}
.m{max-width:85%;padding:8px 12px;border-radius:10px;font-size:14px;line-height:1.5;white-space:pre-wrap}
.m.u{align-self:flex-end;background:var(--accsoft);color:var(--ink)}
.m.a{align-self:flex-start;background:var(--g);border:1px solid var(--line)}
.m .t{display:block;margin-top:6px;font-family:var(--mono);font-size:10.5px;color:var(--acc)}
.row{display:flex;gap:8px}
textarea{flex:1;resize:none;height:54px;padding:9px;font-size:14px;font-family:inherit;
background:var(--g);color:var(--ink);border:1px solid var(--line);border-radius:8px}
button{padding:0 18px;background:var(--acc);color:#fff;border:0;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
.foot{margin-top:14px;font-size:11.5px;color:var(--sub)}.foot a{color:var(--acc)}
</style></head><body><div class="wrap">
<h1>The law, live</h1>
<div class="sub">{{MODEL}} · {{PLACEMENT}} · every reply below is scored against the pre-registered prediction</div>
<div class="law">tok/s = η(tier) × bandwidth ÷ active-bytes-per-token &nbsp;→&nbsp; predicted {{PRED}} tok/s for this box</div>
<div class="cols">
<div>
<div class="panel"><div class="k">Prediction vs reality</div>
  <div class="gauge">
    <div><div class="n" id="pred">{{PRED}}</div><div class="lab">predicted tok/s</div></div>
    <div><div class="n" id="meas">—</div><div class="lab">measured (running avg)</div></div>
    <div class="delta" id="delta"></div>
  </div>
  <div class="log" id="log">each reply appends a data point…</div>
</div>
<div class="panel" style="margin-top:14px"><div class="k">Placement (planned)</div>{{TIERS}}
<div class="foot">Flags: <span style="font-family:var(--mono)">{{FLAGS}}</span></div></div>
<div class="panel" style="margin-top:14px"><div class="k">About</div>
<div class="foot">This page is a running experiment: the tiered decode law predicted this machine's speed
before the model loaded. Laws, probes, evidence: <a href="https://github.com/FedericoTs/quantprobe">repo</a>
· <a href="https://x.com/federico_sciuca">@federico_sciuca</a></div></div>
</div>
<div class="panel chat"><div class="k">Chat — each reply is a measurement</div>
  <div class="msgs" id="msgs"></div>
  <div class="row"><textarea id="inp" placeholder="Say something… (Enter to send)"></textarea>
  <button id="send">Send</button></div>
</div>
</div></div>
<script>
const hist=[],pts=[];const $=id=>document.getElementById(id);
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
async function send(){
  const t=$("inp").value.trim(); if(!t)return;
  $("inp").value=""; $("send").disabled=true;
  hist.push({role:"user",content:t});
  $("msgs").insertAdjacentHTML("beforeend",'<div class="m u">'+esc(t)+'</div>');
  const hold=document.createElement("div");hold.className="m a";hold.textContent="…";
  $("msgs").appendChild(hold);$("msgs").scrollTop=1e9;
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({messages:hist.slice(-12),max_tokens:256})});
    const j=await r.json();
    const msg=(j.choices&&j.choices[0].message.content)||"(no reply)";
    const tps=j.timings?j.timings.predicted_per_second:null;
    hist.push({role:"assistant",content:msg});
    hold.innerHTML=esc(msg)+(tps?'<span class="t">'+tps.toFixed(1)+' tok/s · predicted {{PRED}}</span>':'');
    if(tps){pts.push(tps);
      const avg=pts.reduce((a,b)=>a+b,0)/pts.length,pred=parseFloat("{{PRED}}");
      $("meas").textContent=avg.toFixed(1);
      const d=(avg/pred-1)*100;
      $("delta").textContent=(d>=0?"+":"")+d.toFixed(0)+"%";
      $("delta").style.color=Math.abs(d)<25?"var(--acc)":"var(--warn)";
      $("log").insertAdjacentHTML("afterbegin","#"+pts.length+": "+tps.toFixed(1)+" tok/s<br>");}
  }catch(e){hold.textContent="error: "+e}
  $("send").disabled=false;$("msgs").scrollTop=1e9;
}
$("send").onclick=send;
$("inp").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
</script></body></html>"""


def make_handler(upstream, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            body.setdefault("timings_per_token", True)
            req = urllib.request.Request(upstream + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(502); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
    return H


def dashboard(a):
    best, flags = runtime.best_flags(a)
    binp = runtime.find_llama(a.llama_dir, "llama-server")
    sport = a.server_port
    cmd = [binp, "-m", a.gguf, "--port", str(sport)] + flags
    print(f"[quantprobe] placement: {best[0]}  (predicted {best[1]:.1f} tok/s)")
    print("[quantprobe] llama-server:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    upstream = f"http://127.0.0.1:{sport}"
    for _ in range(120):
        try:
            urllib.request.urlopen(upstream + "/health", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("llama-server exited during startup — check the model path/flags")
            time.sleep(1)
    tiers = ""
    for name, cap, used in runtime.tier_view(a, best):
        pct = min(100, used / cap * 100) if cap else 0
        tiers += (f'<div class="tier"><div class="l"><span>{name}</span>'
                  f'<span>{used:.1f} / {cap:.0f} GB</span></div>'
                  f'<div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div></div>')
    page = (DASH_HTML.replace("{{MODEL}}", os.path.basename(a.gguf))
            .replace("{{PLACEMENT}}", best[0]).replace("{{PRED}}", f"{best[1]:.1f}")
            .replace("{{FLAGS}}", " ".join(flags)).replace("{{TIERS}}", tiers))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(upstream, page))
    url = f"http://127.0.0.1:{a.port}"
    print(f"[quantprobe] dashboard: {url}  (Ctrl-C stops both)")
    if not a.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
