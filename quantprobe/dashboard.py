"""quantprobe dashboard — the law, live.

One command: plans the best placement, launches llama-server with those flags, and serves a local
single-viewport app (no page scrolling: fixed sidebar, ChatGPT-style internal chat scroll) where
every reply is scored against the law's prediction. The NEURON GALAXY renders every expert of every
layer as a dot, illuminated per generated token — thinking tokens in violet, answer tokens in the
color of the memory tier the experts live on.

Honesty rule, printed on the panel: stock llama.cpp exposes no per-token router telemetry, so
expert illumination is sampled uniformly — which Law 2's measured flat routing (Jaccard 1.00)
makes the statistically exact picture. Topic-affinity atlases require an engine that exports
router data; this page shows nothing it cannot back.
"""
from __future__ import annotations
import json, os, subprocess, threading, time, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import runtime

DASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>quantprobe — the law, live</title>
<style>
:root{--g:#fafaf7;--p:#fff;--ink:#16181d;--sub:#5c6066;--line:#e5e4df;--acc:#0f766e;
--accsoft:#e6f2f0;--warn:#b45309;--ram:#d97706;--disk:#dc2626;--think:#7c6bd1;
--mono:Consolas,ui-monospace,monospace}
@media(prefers-color-scheme:dark){:root{--g:#101216;--p:#181b20;--ink:#e8e7e2;--sub:#9aa0a8;
--line:#2b2f35;--acc:#35b8a6;--accsoft:#12332f;--warn:#e0a458;--ram:#f0a94b;--disk:#f26d6d;--think:#a99af0}}
*{box-sizing:border-box}
body{margin:0;height:100vh;overflow:hidden;background:var(--g);color:var(--ink);
font-family:"Segoe UI",system-ui,sans-serif;display:grid;grid-template-rows:auto 1fr}
header{display:flex;align-items:baseline;gap:14px;padding:10px 18px;border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0}
header .sub{color:var(--sub);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
header .law{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--acc);white-space:nowrap}
main{display:grid;grid-template-columns:320px 1fr;gap:14px;padding:14px;min-height:0}
aside{overflow-y:auto;min-height:0;display:flex;flex-direction:column;gap:12px}
.panel{background:var(--p);border:1px solid var(--line);border-radius:10px;padding:12px;flex-shrink:0}
.k{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--sub);font-weight:600;margin-bottom:7px}
.gauge{display:flex;gap:14px;align-items:baseline}
.gauge .n{font-family:var(--mono);font-size:34px;font-weight:700;color:var(--acc);font-variant-numeric:tabular-nums}
.gauge .lab{font-size:10.5px;color:var(--sub)}
.delta{font-family:var(--mono);font-size:14px;font-weight:600}
.log{font-family:var(--mono);font-size:11px;color:var(--sub);max-height:72px;overflow-y:auto;margin-top:6px}
.stat{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11.5px;color:var(--sub);margin:3px 0}
.stat b{color:var(--ink);font-weight:600}
.tgl{display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer;user-select:none}
.tgl input{accent-color:var(--think);width:15px;height:15px}
.tier{margin:7px 0}.tier .l{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;color:var(--sub);margin-bottom:3px}
.bar{height:10px;background:var(--g);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--acc)}
.foot{font-size:11px;color:var(--sub);line-height:1.5}.foot a{color:var(--acc)}
.right{display:grid;grid-template-rows:minmax(220px,37%) 1fr;gap:14px;min-height:0}
.brainp{position:relative;background:var(--p);border:1px solid var(--line);border-radius:10px;padding:10px 12px;
display:flex;flex-direction:column;min-height:0}
#brain{flex:1;width:100%;min-height:0;border-radius:6px}
.legend{display:flex;gap:13px;font-size:10.5px;color:var(--sub);margin-top:6px;flex-wrap:wrap;align-items:center}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:-1px}
.chatp{display:flex;flex-direction:column;min-height:0;background:var(--p);border:1px solid var(--line);border-radius:10px;padding:12px}
.msgs{flex:1;overflow-y:auto;min-height:0;display:flex;flex-direction:column;gap:8px;padding-right:4px}
.m{max-width:85%;padding:8px 12px;border-radius:10px;font-size:13.5px;line-height:1.5;white-space:pre-wrap}
.m.u{align-self:flex-end;background:var(--accsoft)}
.m.a{align-self:flex-start;background:var(--g);border:1px solid var(--line)}
.m .t{display:block;margin-top:6px;font-family:var(--mono);font-size:10px;color:var(--acc)}
.m pre{background:var(--p);border:1px solid var(--line);border-radius:6px;padding:8px;overflow-x:auto;font-family:var(--mono);font-size:12px;margin:6px 0;white-space:pre}
.thk{align-self:flex-start;max-width:85%;font-size:11.5px;color:var(--think);border-left:3px solid var(--think);
padding:3px 10px;opacity:.85;white-space:pre-wrap}
.thk span{display:block;max-height:80px;overflow-y:auto}
.thk summary{cursor:pointer;font-size:10.5px;letter-spacing:.05em}
.row{display:flex;gap:8px;margin-top:10px}
textarea{flex:1;resize:none;height:52px;padding:9px;font-size:13.5px;font-family:inherit;
background:var(--g);color:var(--ink);border:1px solid var(--line);border-radius:8px}
button{padding:0 18px;background:var(--acc);color:#fff;border:0;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
</style></head><body>
<header><h1>The law, live</h1>
<div class="sub">{{MODEL}} · {{PLACEMENT}}</div>
<div class="law">tok/s = η·BW ÷ active-bytes → predicted {{PRED}}</div></header>
<main>
<aside>
  <div class="panel"><div class="k">Prediction vs reality</div>
    <div class="gauge">
      <div><div class="n">{{PRED}}</div><div class="lab">predicted tok/s</div></div>
      <div><div class="n" id="meas">—</div><div class="lab">measured avg</div></div>
      <div class="delta" id="delta"></div>
    </div>
    <div class="log" id="log">each reply appends a data point…</div>
  </div>
  <div class="panel"><div class="k">Thinking</div>
    <label class="tgl"><input type="checkbox" id="think" checked> allow thinking <span style="color:var(--think)">●</span></label>
    <div style="margin-top:8px">
      <div class="stat"><span>time to first token</span><b id="ttft">—</b></div>
      <div class="stat"><span>thinking</span><b id="sthink">—</b></div>
      <div class="stat"><span>answer</span><b id="sans">—</b></div>
    </div>
    <div class="foot" style="margin-top:6px">Thinking improves hard answers but costs latency at
    decode speed — toggle it off for quick tasks.</div>
  </div>
  <div class="panel"><div class="k">Placement (planned)</div>{{TIERS}}
    <div class="foot" style="font-family:var(--mono);font-size:10px">{{FLAGS}}</div></div>
  <div class="panel"><div class="k">About</div>
    <div class="foot">A running experiment: the law predicted this machine's speed before the model
    loaded. <a href="https://github.com/FedericoTs/quantprobe">repo</a> ·
    <a href="https://x.com/federico_sciuca">@federico_sciuca</a></div></div>
</aside>
<div class="right">
  <div class="brainp"><div class="k">Neuron galaxy — {{NL}} layers × {{NE}} experts · every token traverses all layers</div>
    <canvas id="brain"></canvas>
    <div class="legend">
      <span><span class="dot" style="background:var(--acc)"></span>attention + KV → {{ATIER}}</span>
      <span><span class="dot" style="background:var({{ECOLV}})"></span>experts ({{NK}} of {{NE}} fire per layer per token) → {{ETIER}}</span>
      <span><span class="dot" style="background:var(--think)"></span>thinking</span>
      <span style="opacity:.75">flashes sampled uniformly — statistically exact under measured flat routing (Jaccard 1.00); stock llama.cpp exposes no router telemetry</span>
    </div>
  </div>
  <div class="chatp"><div class="k">Chat — each reply is a measurement</div>
    <div class="msgs" id="msgs"></div>
    <div class="row"><textarea id="inp" placeholder="Say something… (Enter to send)"></textarea>
    <button id="send">Send</button></div>
  </div>
</div>
</main>
<script>
const NL={{NL}},NE={{NE}},NK={{NK}},PRED=parseFloat("{{PRED}}");
const hist=[],pts=[];const $=id=>document.getElementById(id);
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function md(s){return esc(s).replace(/```(\\w*)\\n?([\\s\\S]*?)```/g,'<pre>$2</pre>')}
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||"#888"}

/* ---- neuron galaxy ---- */
const cv=$("brain"),cx2=cv.getContext("2d");
const NDOT=Math.max(NE,1);let W,H,px,py,ax,ay,heat,tint,aheat;
function layout(){
  W=cv.width=cv.clientWidth*devicePixelRatio;H=cv.height=cv.clientHeight*devicePixelRatio;
  const CX=W/2,CY=H/2,MR=Math.min(W,H)*0.46;
  px=new Float32Array(NL*NDOT);py=new Float32Array(NL*NDOT);
  ax=new Float32Array(NL);ay=new Float32Array(NL);
  heat=new Float32Array(NL*NDOT);tint=new Uint8Array(NL*NDOT);aheat=new Float32Array(NL);
  for(let l=0;l<NL;l++){
    const t=NL>1?l/(NL-1):0, ang=t*Math.PI*3.8+0.7, R=(0.18+0.78*t)*MR;
    const cxl=CX+R*Math.cos(ang), cyl=CY+R*Math.sin(ang)*0.6;
    ax[l]=cxl;ay[l]=cyl;
    const sig=MR*(0.05+0.05*Math.sin(l*3.7)*Math.sin(l*3.7));
    for(let e=0;e<NDOT;e++){
      const r=sig*Math.sqrt((e+0.5)/NDOT)*2.1, a=e*2.39996+l*0.7;
      px[l*NDOT+e]=cxl+r*Math.cos(a);
      py[l*NDOT+e]=cyl+r*Math.sin(a)*0.85;
    }
  }
  draw();
}
function draw(){
  cx2.clearRect(0,0,W,H);
  const ec=cssv("{{ECOLV}}"),tc=cssv("--think"),ac=cssv("--acc");
  const s=2.1*devicePixelRatio, sa=3.4*devicePixelRatio;
  for(let i=0;i<NL*NDOT;i++){
    const h=heat[i];
    cx2.globalAlpha=0.09+0.91*h;
    cx2.fillStyle=(h>0.03&&tint[i])?tc:ec;
    cx2.fillRect(px[i],py[i],s,s);
  }
  for(let l=0;l<NL;l++){
    cx2.globalAlpha=0.35+0.65*aheat[l];
    cx2.fillStyle=ac;
    cx2.beginPath();cx2.arc(ax[l],ay[l],sa,0,7);cx2.fill();
  }
  cx2.globalAlpha=1;
}
let decaying=false;
function decay(){let live=false;
  for(let i=0;i<heat.length;i++){heat[i]*=0.86;if(heat[i]>0.02)live=true}
  for(let l=0;l<NL;l++){aheat[l]*=0.86;if(aheat[l]>0.02)live=true}
  draw();
  if(live)requestAnimationFrame(decay);else decaying=false}
function pulse(thinking){
  for(let l=0;l<NL;l++){aheat[l]=1;
    for(let k=0;k<Math.max(NK,1);k++){const i=l*NDOT+((Math.random()*NDOT)|0);
      heat[i]=1;tint[i]=thinking?1:0}}
  if(!decaying){decaying=true;requestAnimationFrame(decay)}}
addEventListener("resize",layout);layout();

/* ---- streaming chat with reply anatomy ---- */
function fmt(ms){return ms<1000?ms.toFixed(0)+" ms":(ms/1000).toFixed(1)+" s"}
async function send(){
  const t=$("inp").value.trim(); if(!t)return;
  $("inp").value=""; $("send").disabled=true;
  hist.push({role:"user",content:t});
  $("msgs").insertAdjacentHTML("beforeend",'<div class="m u">'+esc(t)+'</div>');
  const thk=document.createElement("details");thk.className="thk";thk.open=true;
  thk.innerHTML='<summary>thinking…</summary><span></span>';
  const hold=document.createElement("div");hold.className="m a";hold.textContent="…";
  $("msgs").appendChild(thk);$("msgs").appendChild(hold);$("msgs").scrollTop=1e9;
  let content="",think="",nT=0,nA=0,t0=performance.now(),tFirst=0,tThinkEnd=0,timings=null;
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({messages:hist.slice(-12),max_tokens:2048,stream:true,think:$("think").checked})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
    while(true){
      const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf("\\n"))>=0){
        const line=buf.slice(0,i).trim();buf=buf.slice(i+1);
        if(!line.startsWith("data:"))continue;
        const p=line.slice(5).trim(); if(p==="[DONE]")continue;
        let j;try{j=JSON.parse(p)}catch(e){continue}
        if(j.timings)timings=j.timings;
        const d=(j.choices&&j.choices[0]&&j.choices[0].delta)||{};
        if(d.reasoning_content){think+=d.reasoning_content;nT++;pulse(true);
          if(!tFirst)tFirst=performance.now();
          thk.lastElementChild.textContent=think.slice(-1500);
          thk.lastElementChild.scrollTop=1e9}
        if(d.content){content+=d.content;nA++;pulse(false);
          if(!tFirst)tFirst=performance.now();
          if(!tThinkEnd)tThinkEnd=performance.now();
          hold.innerHTML=md(content)}
      }
      $("msgs").scrollTop=1e9;
    }
  }catch(e){hold.textContent="error: "+e}
  const tEnd=performance.now();
  if(think){thk.open=false;
    thk.firstElementChild.textContent="thought for ~"+think.split(/\\s+/).length+" words (click to expand)"}
  else thk.remove();
  if(!content)content="(empty reply — try again)";
  hist.push({role:"assistant",content:content});
  const tps=timings?timings.predicted_per_second:((nT+nA)>0?(nT+nA)/((tEnd-t0)/1000):0);
  hold.innerHTML=md(content)+'<span class="t">'+tps.toFixed(1)+' tok/s'+(timings?"":" (client est.)")+' · predicted '+PRED+'</span>';
  $("ttft").textContent=tFirst?fmt(tFirst-t0):"—";
  $("sthink").textContent=nT?fmt((tThinkEnd||tEnd)-(tFirst||t0))+" · "+nT+" tok":"off";
  $("sans").textContent=nA?fmt(tEnd-(tThinkEnd||tFirst||t0))+" · "+nA+" tok":"—";
  if(tps>0){pts.push(tps);
    const avg=pts.reduce((a,b)=>a+b,0)/pts.length;
    $("meas").textContent=avg.toFixed(1);
    const dd=(avg/PRED-1)*100;
    $("delta").textContent=(dd>=0?"+":"")+dd.toFixed(0)+"%";
    $("delta").style.color=Math.abs(dd)<25?"var(--acc)":"var(--warn)";
    $("log").insertAdjacentHTML("afterbegin","#"+pts.length+": "+tps.toFixed(1)+" tok/s<br>");}
  $("send").disabled=false;$("msgs").scrollTop=1e9;
}
$("send").onclick=send;
$("inp").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
</script></body></html>"""


def make_handler(upstream, page):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            body.setdefault("timings_per_token", True)
            body.setdefault("max_tokens", 2048)
            if body.pop("think", True) is False:      # UI toggle: skip thinking entirely
                body["chat_template_kwargs"] = {"enable_thinking": False}
            stream = bool(body.get("stream", False))
            req = urllib.request.Request(upstream + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                r = urllib.request.urlopen(req, timeout=600)
                self.send_response(200)
                if stream:
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = r.read1(8192) if hasattr(r, "read1") else r.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    data = r.read()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except Exception as e:
                try:
                    err = json.dumps({"error": str(e)}).encode()
                    self.send_response(502)
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                except Exception:
                    pass
    return H


def _anatomy(gguf):
    """Exact layer/expert counts from the file, for the galaxy."""
    try:
        from gguf import GGUFReader
        r = GGUFReader(gguf)
        nl = ne = nk = 0
        for f in r.fields.values():
            if f.name.endswith(".block_count"):
                nl = int(f.parts[f.data[0]][0])
            elif f.name.endswith(".expert_count"):
                ne = int(f.parts[f.data[0]][0])
            elif f.name.endswith(".expert_used_count"):
                nk = int(f.parts[f.data[0]][0])
        return (nl or 32), ne, (nk if ne else 0)
    except Exception:
        return 32, 0, 0


def dashboard(a):
    best, flags = runtime.best_flags(a)
    binp = runtime.find_llama(a.llama_dir, "llama-server")
    sport = a.server_port
    cmd = [binp, "-m", a.gguf, "--port", str(sport)] + flags
    print(f"[quantprobe] placement: {best[0]}  (predicted {best[1]:.1f} tok/s)")
    print("[quantprobe] llama-server:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    upstream = f"http://127.0.0.1:{sport}"
    print("[quantprobe] waiting for the model to finish loading (big models on slow disks take minutes)...")
    ready_req = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    for _ in range(600):
        try:
            urllib.request.urlopen(urllib.request.Request(
                upstream + "/v1/chat/completions", data=ready_req,
                headers={"Content-Type": "application/json"}), timeout=8)
            break                                   # a real completion answered: weights are up
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("llama-server exited during startup — check the model path/flags")
            time.sleep(2)
    tiers = ""
    for name, cap, used in runtime.tier_view(a, best):
        pct = min(100, used / cap * 100) if cap else 0
        tiers += (f'<div class="tier"><div class="l"><span>{name}</span>'
                  f'<span>{used:.1f} / {cap:.0f} GB</span></div>'
                  f'<div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div></div>')
    nl, ne, nk = _anatomy(a.gguf)
    pname = best[0]
    if "hybrid" in pname:
        atier, etier, ecolv = "VRAM", "RAM", "--ram"
    elif "all in VRAM" in pname:
        atier, etier, ecolv = "VRAM", "VRAM", "--acc"
    elif "disk" in pname:
        atier, etier, ecolv = "RAM", "disk (streamed, RAM-cached)", "--disk"
    else:
        atier, etier, ecolv = "RAM", "RAM", "--ram"
    page = (DASH_HTML.replace("{{MODEL}}", os.path.basename(a.gguf))
            .replace("{{PLACEMENT}}", pname).replace("{{PRED}}", f"{best[1]:.1f}")
            .replace("{{FLAGS}}", " ".join(flags)).replace("{{TIERS}}", tiers)
            .replace("{{NL}}", str(nl)).replace("{{NE}}", str(ne)).replace("{{NK}}", str(nk))
            .replace("{{ATIER}}", atier).replace("{{ETIER}}", etier).replace("{{ECOLV}}", ecolv))
    ThreadingHTTPServer.allow_reuse_address = False   # Windows: reuse allows silent double-binds
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(upstream, page))
    except OSError:
        proc.terminate()
        raise SystemExit(f"port {a.port} is already in use - another dashboard running? (pass --port N)")
    url = f"http://127.0.0.1:{a.port}"
    print(f"[quantprobe] dashboard v2.1: {url}  (Ctrl-C stops both)")
    if not a.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
