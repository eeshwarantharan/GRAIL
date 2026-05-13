from __future__ import annotations

from pathlib import Path
import base64
import json
import math

from gnss_vim_sim.viz.plots import load_rows
from gnss_vim_sim.world.scene import MeshScene


def _num(rows: list[dict], key: str, default=0.0) -> list[float]:
    out = []
    for row in rows:
        val = row.get(key, default)
        out.append(float(default if val == "" else val))
    return out


def _num_first(rows: list[dict], keys: list[str], default=0.0) -> list[float]:
    for key in keys:
        if rows and key in rows[0]:
            return _num(rows, key, default)
    return [float(default) for _ in rows]


def _bool(rows: list[dict], key: str) -> list[int]:
    return [1 if row.get(key) is True else 0 for row in rows]


def _data_uri(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def make_flight_player(
    log_path: Path,
    out_html: Path,
    scene: MeshScene | None = None,
    drone_icon: Path | None = None,
    fps: int = 30,
) -> bool:
    rows = load_rows(log_path)
    if not rows:
        return False
    scene = scene or MeshScene(None)
    route_stride = max(1, len(rows) // 5000)
    trail_stride = max(1, len(rows) // 1800)
    payload = {
        "fps": fps,
        "mesh": scene.webgl_mesh(max_faces=80_000),
        "bounds": scene.stats().__dict__,
        "t": _num(rows, "t"),
        "x": _num(rows, "true_x"),
        "y": _num(rows, "true_y"),
        "z": _num(rows, "true_z"),
        "risk": _num_first(rows, ["model_score", "ml_risk"]),
        "bad": _bool(rows, "gnss_bad_truth"),
        "gnss": _num(rows, "gnss_z", math.nan),
        "ml": _num(rows, "ml_integrity_z"),
        "vdop": _num(rows, "gnss_vdop", math.nan),
        "range": _bool(rows, "ml_integrity_range_fired"),
        "route": [
            [
                float(rows[i]["true_x"]),
                float(rows[i]["true_y"]),
                float(rows[i]["true_z"]),
                float(rows[i].get("ml_risk", 0.0)),
                1 if rows[i].get("gnss_bad_truth") is True else 0,
            ]
            for i in range(0, len(rows), route_stride)
        ],
        "trail": [
            [float(rows[i]["true_x"]), float(rows[i]["true_y"]), float(rows[i]["true_z"]), float(rows[i]["ml_risk"])]
            for i in range(0, len(rows), trail_stride)
        ],
        "icon": _data_uri(drone_icon),
    }
    html = _HTML.replace("__PAYLOAD__", json.dumps(payload, allow_nan=True))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return True


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GNSS-VIM WebGL Flight Player</title>
<style>
  :root{--bg:#eef3f8;--panel:#ffffff;--panel2:#f8fafc;--muted:#64748b;--line:#cbd5e1;--blue:#0284c7;--ink:#102033}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,Segoe UI,sans-serif;overflow:hidden}
  .app{height:100vh;display:grid;grid-template-columns:minmax(720px,1fr) 420px;gap:10px;padding:10px}
  .left{display:grid;grid-template-rows:1fr 250px;gap:10px;min-width:0}
  .side{display:grid;grid-template-rows:1fr 1fr 1fr;gap:10px;min-width:0}
  .panel{position:relative;overflow:hidden;background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 10px 28px #0f172a18}
  canvas{display:block;position:absolute;inset:0;width:100%;height:100%}
  .hud{position:absolute;left:16px;top:14px;background:#fffffff0;border:1px solid var(--line);border-radius:10px;padding:10px 12px;backdrop-filter:blur(8px);z-index:4;box-shadow:0 8px 22px #0f172a18}
  .hud b{font-size:18px}.hud span{display:block;color:var(--muted);font-size:12px;margin-top:3px}
  .toolbar{position:absolute;right:14px;top:14px;display:flex;gap:8px;z-index:5}
  .controls{position:absolute;left:16px;right:16px;bottom:14px;display:grid;grid-template-columns:86px 86px 178px 1fr 70px;gap:10px;align-items:center;z-index:5}
  button{background:#0f6ea8;border:1px solid #0284c7;color:white;border-radius:8px;padding:9px 10px;font-weight:800;cursor:pointer}
  button.secondary{background:#f8fafc;color:#102033;border-color:#cbd5e1}
  .speed{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}
  .speed button{padding:8px 0}
  .speed button.active{background:#0f6ea8;color:white;border-color:#0284c7}
  input[type=range]{width:100%;accent-color:#0284c7}
  .title{position:absolute;left:14px;top:12px;color:#0f2742;font-weight:850;z-index:3}
  .legend{position:absolute;right:12px;top:12px;color:var(--muted);font-size:12px;text-align:right;z-index:3}
  .metric{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:46px 14px 14px}
  .box{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px}
  .box label{color:var(--muted);font-size:12px}.box div{font-size:22px;font-weight:850;margin-top:4px}
  .key{position:absolute;left:14px;bottom:12px;color:#334155;font-size:12px;background:#ffffffd9;border:1px solid var(--line);border-radius:8px;padding:7px 9px;z-index:4}
  .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px}.sw:first-child{margin-left:0}
</style>
</head>
<body>
<div class="app">
  <div class="left">
    <div class="panel" id="stage">
      <canvas id="gl"></canvas><canvas id="overlay"></canvas>
      <div class="hud"><b id="mode">CHASE 45</b><span id="hud"></span></div>
      <div class="toolbar"><button class="secondary" id="chase">Chase 45</button><button class="secondary" id="top">Top</button><button class="secondary" id="wide">Wide</button></div>
      <div class="key"><span class="sw" style="background:#16a34a"></span>clean <span class="sw" style="background:#f59e0b"></span>high ML risk <span class="sw" style="background:#dc2626"></span>bad GNSS epoch</div>
      <div class="controls"><button id="play">Play</button><button id="reset">Reset</button><div class="speed"><button class="secondary active" data-speed="1">1x</button><button class="secondary" data-speed="2">2x</button><button class="secondary" data-speed="4">4x</button><button class="secondary" data-speed="8">8x</button></div><input id="scrub" type="range" min="0" max="1" step="0.0001" value="0"><span id="clock">0.0s</span></div>
    </div>
    <div class="panel"><canvas id="plot"></canvas><div class="title">Altitude Fusion</div><div class="legend">black truth · gray GNSS · blue ML</div></div>
  </div>
  <div class="side">
    <div class="panel" id="mapStage"><canvas id="mapgl"></canvas><canvas id="mapov"></canvas><div class="title">WebGL Top-Down Mesh</div><div class="legend">same route colors as main view</div></div>
    <div class="panel"><canvas id="risk"></canvas><div class="title">GNSS Vertical Integrity</div><div class="legend">purple risk · orange VDOP · red range aid</div></div>
    <div class="panel"><div class="title">Live Telemetry (Read-Only)</div><div class="metric">
      <div class="box"><label>ML risk</label><div id="mRisk">0.00</div></div>
      <div class="box"><label>Altitude</label><div id="mAlt">0.0m</div></div>
      <div class="box"><label>VDOP</label><div id="mVdop">0.0</div></div>
      <div class="box"><label>Range aid</label><div id="mRange">OFF</div></div>
      <div class="box"><label>GNSS epoch</label><div id="mGnssStatus">OK</div></div>
      <div class="box"><label>Playback</label><div id="mSpeed">1x</div></div>
      <div class="box"><label>View</label><div id="mView">Chase</div></div>
      <div class="box"><label>Mesh</label><div id="mMesh">0</div></div>
    </div></div>
  </div>
</div>
<script>
const D=__PAYLOAD__, mesh=D.mesh||{positions:[],normals:[],indices:[],mode:"empty"};
const img=new Image(); if(D.icon) img.src=D.icon;
let playing=false, simT=D.t[0], last=performance.now(), speed=1.0, viewMode="chase";
const minT=D.t[0], maxT=D.t[D.t.length-1], $=id=>document.getElementById(id);
$("mMesh").textContent=mesh.mode==="triangles"?Math.floor(mesh.indices.length/3)+"f":Math.floor(mesh.positions.length/3)+"v";
$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play"};
$("reset").onclick=()=>{simT=minT;playing=false;$("play").textContent="Play"};
$("scrub").oninput=e=>{simT=minT+(maxT-minT)*parseFloat(e.target.value)};
document.querySelectorAll(".speed button").forEach(b=>b.onclick=()=>{speed=parseFloat(b.dataset.speed);$("mSpeed").textContent=speed+"x";document.querySelectorAll(".speed button").forEach(x=>x.classList.remove("active"));b.classList.add("active")});
$("chase").onclick=()=>{viewMode="chase";$("mode").textContent="CHASE 45";$("mView").textContent="Chase"};
$("top").onclick=()=>{viewMode="top";$("mode").textContent="TOP DOWN";$("mView").textContent="Top"};
$("wide").onclick=()=>{viewMode="wide";$("mode").textContent="WIDE 45";$("mView").textContent="Wide"};
function idxAt(t){let lo=0,hi=D.t.length-1;while(lo<hi){let m=(lo+hi)>>1;if(D.t[m]<t)lo=m+1;else hi=m}return Math.max(0,Math.min(D.t.length-1,lo))}
function lerp(a,b,u){return a+(b-a)*u}
function sample(t){let i=idxAt(t),j=Math.max(0,i-1),u=(t-D.t[j])/Math.max(D.t[i]-D.t[j],1e-6);return {i,x:lerp(D.x[j],D.x[i],u),y:lerp(D.y[j],D.y[i],u),z:lerp(D.z[j],D.z[i],u),risk:lerp(D.risk[j],D.risk[i],u)}}
function heading(i){let j=Math.min(D.x.length-1,i+18),k=Math.max(0,i-18);let a=Math.atan2(D.y[j]-D.y[k],D.x[j]-D.x[k]);return isFinite(a)?a:0}
function riskColor(r,bad=false){return bad?"#dc2626":r>=0.55?"#f59e0b":"#16a34a"}
function fit2(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;if(c.width!==r.width*d||c.height!==r.height*d){c.width=r.width*d;c.height=r.height*d}const ctx=c.getContext("2d");ctx.setTransform(d,0,0,d,0,0);return [ctx,r.width,r.height]}
function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]}function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function muls(a,s){return [a[0]*s,a[1]*s,a[2]*s]}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function norm(a){let l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l]}
function perspective(fovy,aspect,near,far){let f=1/Math.tan(fovy/2),nf=1/(near-far);return [f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]}
function lookAt(e,c,up){let z=norm(sub(e,c)),x=norm(cross(up,z)),y=cross(z,x);return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,e),-dot(y,e),-dot(z,e),1]}
function matMul(a,b){let o=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
let routeMinX=Math.min(...D.x),routeMaxX=Math.max(...D.x),routeMinY=Math.min(...D.y),routeMaxY=Math.max(...D.y);
let routeSpan=Math.max(routeMaxX-routeMinX,routeMaxY-routeMinY,60);
function cameraFor(s){let a=heading(s.i),f=[Math.cos(a),Math.sin(a),0],pos=[s.x,s.y,s.z];if(viewMode==="top"){return {eye:[s.x,s.y,s.z+routeSpan*1.25],target:[s.x,s.y,s.z],fov:35*Math.PI/180}}if(viewMode==="wide"){return {eye:[s.x-routeSpan*.55,s.y-routeSpan*.55,s.z+routeSpan*.45],target:[s.x,s.y,s.z+8],fov:48*Math.PI/180}}return {eye:add(sub(pos,muls(f,72)),[0,0,42]),target:add(add(pos,muls(f,36)),[0,0,8]),fov:48*Math.PI/180}}
function camUp(cam){let f=norm(sub(cam.target,cam.eye));return Math.abs(f[2])>0.94?[0,1,0]:[0,0,1]}
function project(p,cam,w,h){let e=cam.eye,t=cam.target,bf=norm(sub(t,e)),br=norm(cross(bf,camUp(cam))),bu=norm(cross(br,bf)),rel=sub(p,e),z=dot(rel,bf);if(z<=0)return null;let tan=Math.tan(cam.fov/2),asp=w/h;return [w/2+(dot(rel,br)/(z*tan*asp))*w/2,h/2-(dot(rel,bu)/(z*tan))*h/2,z]}

const glCanvas=$("gl"), ov=$("overlay"), stage=$("stage"), gl=glCanvas.getContext("webgl",{antialias:true});
function makeGLKit(g){
 function shader(type,src){let s=g.createShader(type);g.shaderSource(s,src);g.compileShader(s);if(!g.getShaderParameter(s,g.COMPILE_STATUS))throw new Error(g.getShaderInfoLog(s));return s}
 function program(vs,fs){let p=g.createProgram();g.attachShader(p,shader(g.VERTEX_SHADER,vs));g.attachShader(p,shader(g.FRAGMENT_SHADER,fs));g.linkProgram(p);if(!g.getProgramParameter(p,g.LINK_STATUS))throw new Error(g.getProgramInfoLog(p));return p}
 function buf(data){let b=g.createBuffer();g.bindBuffer(g.ARRAY_BUFFER,b);g.bufferData(g.ARRAY_BUFFER,new Float32Array(data),g.STATIC_DRAW);return {b,count:data.length/3}}
 const meshProg=program(`attribute vec3 aPos;attribute vec3 aNor;uniform mat4 uMvp;varying vec3 vNor;varying float vZ;void main(){vNor=aNor;vZ=aPos.z;gl_PointSize=2.4;gl_Position=uMvp*vec4(aPos,1.0);}`,`precision mediump float;varying vec3 vNor;varying float vZ;void main(){vec3 n=normalize(vNor+vec3(0.0,0.0,0.001));float l=max(dot(n,normalize(vec3(-0.45,-0.3,0.84))),0.0);vec3 ground=vec3(0.72,0.77,0.82);vec3 wall=vec3(0.48,0.55,0.63);vec3 c=mix(ground,wall,clamp(vZ/34.0,0.0,1.0))*(0.72+0.42*l);gl_FragColor=vec4(c,1.0);}`);
 const lineProg=program(`attribute vec3 aPos;uniform mat4 uMvp;void main(){gl_Position=uMvp*vec4(aPos,1.0);}`,`precision mediump float;uniform vec4 uColor;void main(){gl_FragColor=uColor;}`);
 const posBuf=buf(mesh.positions), norBuf=buf(mesh.normals.length?mesh.normals:new Array(mesh.positions.length).fill(0).map((_,i)=>i%3===2?1:0));
 let idxBuf=null,idxCount=0,idxType=g.UNSIGNED_SHORT;if(mesh.indices.length){let use32=mesh.positions.length/3>65535;if(use32&&!g.getExtension("OES_element_index_uint"))throw new Error("32-bit mesh indices unsupported");idxBuf=g.createBuffer();g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,idxBuf);g.bufferData(g.ELEMENT_ARRAY_BUFFER,use32?new Uint32Array(mesh.indices):new Uint16Array(mesh.indices),g.STATIC_DRAW);idxCount=mesh.indices.length;idxType=use32?g.UNSIGNED_INT:g.UNSIGNED_SHORT}
 return {g,meshProg,lineProg,buf,posBuf,norBuf,idxBuf,idxCount,idxType};
}
const mainKit=makeGLKit(gl);
const routeFlat=D.route.flatMap(p=>[p[0],p[1],p[2]+0.35]), routeBuf=mainKit.buf(routeFlat);
function gridData(){let d=[],step=20,pad=80,x0=Math.floor((routeMinX-pad)/step)*step,x1=Math.ceil((routeMaxX+pad)/step)*step,y0=Math.floor((routeMinY-pad)/step)*step,y1=Math.ceil((routeMaxY+pad)/step)*step;for(let x=x0;x<=x1;x+=step)d.push(x,y0,0,x,y1,0);for(let y=y0;y<=y1;y+=step)d.push(x0,y,0,x1,y,0);return d}
const gridBuf=mainKit.buf(gridData());
const mapCanvas=$("mapgl"), mapOv=$("mapov"), mapStage=$("mapStage"), mapGl=mapCanvas.getContext("webgl",{antialias:true}), mapKit=makeGLKit(mapGl), mapGridBuf=mapKit.buf(gridData()), mapRouteBuf=mapKit.buf(routeFlat);
function resizeCanvas(canvas, overlay, holder, g){let d=window.devicePixelRatio||1,r=holder.getBoundingClientRect();for(const c of [canvas,overlay]){if(c.width!==r.width*d||c.height!==r.height*d){c.width=r.width*d;c.height=r.height*d;c.style.width=r.width+"px";c.style.height=r.height+"px"}}g.viewport(0,0,canvas.width,canvas.height)}
function resize(){resizeCanvas(glCanvas,ov,stage,gl)}
function drawLines(kit,buffer,count,color,M,mode){let g=kit.g;g.useProgram(kit.lineProg);let a=g.getAttribLocation(kit.lineProg,"aPos");g.bindBuffer(g.ARRAY_BUFFER,buffer.b);g.enableVertexAttribArray(a);g.vertexAttribPointer(a,3,g.FLOAT,false,0,0);g.uniformMatrix4fv(g.getUniformLocation(kit.lineProg,"uMvp"),false,M);g.uniform4fv(g.getUniformLocation(kit.lineProg,"uColor"),color);g.drawArrays(mode??g.LINES,0,count)}
function drawMesh(kit,M){let g=kit.g;g.useProgram(kit.meshProg);let ap=g.getAttribLocation(kit.meshProg,"aPos"),an=g.getAttribLocation(kit.meshProg,"aNor");g.bindBuffer(g.ARRAY_BUFFER,kit.posBuf.b);g.enableVertexAttribArray(ap);g.vertexAttribPointer(ap,3,g.FLOAT,false,0,0);g.bindBuffer(g.ARRAY_BUFFER,kit.norBuf.b);g.enableVertexAttribArray(an);g.vertexAttribPointer(an,3,g.FLOAT,false,0,0);g.uniformMatrix4fv(g.getUniformLocation(kit.meshProg,"uMvp"),false,M);if(kit.idxBuf){g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,kit.idxBuf);g.drawElements(g.TRIANGLES,kit.idxCount,kit.idxType,0)}else{g.drawArrays(g.POINTS,0,kit.posBuf.count)}}
function clearScene(g){g.clearColor(0.965,0.977,0.989,1);g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);g.enable(g.DEPTH_TEST)}
function drawWebGL(s){resize();let cam=cameraFor(s),view=lookAt(cam.eye,cam.target,camUp(cam)),proj=perspective(cam.fov,glCanvas.width/glCanvas.height,0.2,6000),M=new Float32Array(matMul(proj,view));clearScene(gl);drawLines(mainKit,gridBuf,gridBuf.count,new Float32Array([0.15,0.42,0.65,0.34]),M,gl.LINES);drawMesh(mainKit,M);gl.disable(gl.DEPTH_TEST);drawLines(mainKit,routeBuf,routeBuf.count,new Float32Array([0.15,0.23,0.32,0.42]),M,gl.LINE_STRIP);return cam}
function drawRouteOverlay(ctx,w,h,cam,lineWidth=4){ctx.lineCap="round";ctx.lineJoin="round";for(let i=1;i<D.route.length;i++){let a=D.route[i-1],b=D.route[i],pa=project([a[0],a[1],a[2]+0.8],cam,w,h),pb=project([b[0],b[1],b[2]+0.8],cam,w,h);if(!pa||!pb)continue;ctx.strokeStyle=riskColor(Math.max(a[3],b[3]),a[4]||b[4]);ctx.lineWidth=lineWidth;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke()}}
function drawDroneOverlay(ctx,p,ang,size){ctx.save();ctx.translate(p[0],p[1]);ctx.rotate(-ang+Math.PI/2);if(img.complete&&img.width){ctx.drawImage(img,-size/2,-size/2,size,size)}else{ctx.fillStyle="#0f766e";ctx.beginPath();ctx.moveTo(0,-size*.45);ctx.lineTo(size*.32,size*.36);ctx.lineTo(0,size*.18);ctx.lineTo(-size*.32,size*.36);ctx.closePath();ctx.fill()}ctx.restore()}
function drawOverlay(s,cam){let [ctx,w,h]=fit2(ov);ctx.clearRect(0,0,w,h);drawRouteOverlay(ctx,w,h,cam,4);let p=project([s.x,s.y,s.z+2.2],cam,w,h);if(p){drawDroneOverlay(ctx,p,heading(s.i),56);ctx.fillStyle=riskColor(s.risk,D.bad[s.i]);ctx.beginPath();ctx.arc(p[0]+32,p[1]-25,8,0,7);ctx.fill();ctx.strokeStyle="#0f172a";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0]-22,p[1]+34);ctx.lineTo(p[0]+22,p[1]+34);ctx.stroke()}ctx.fillStyle="#334155";ctx.font="12px Inter,system-ui";ctx.fillText(`WebGL ${mesh.mode} mesh | route color = GNSS integrity`,16,58)}
function mapCamera(){let cx=(routeMinX+routeMaxX)/2,cy=(routeMinY+routeMaxY)/2;return {eye:[cx,cy,routeSpan*1.55],target:[cx,cy,0],fov:42*Math.PI/180}}
function drawMapWebGL(s){resizeCanvas(mapCanvas,mapOv,mapStage,mapGl);let cam=mapCamera(),view=lookAt(cam.eye,cam.target,camUp(cam)),proj=perspective(cam.fov,mapCanvas.width/mapCanvas.height,0.2,6000),M=new Float32Array(matMul(proj,view));clearScene(mapGl);drawLines(mapKit,mapGridBuf,mapGridBuf.count,new Float32Array([0.15,0.42,0.65,0.28]),M,mapGl.LINES);drawMesh(mapKit,M);mapGl.disable(mapGl.DEPTH_TEST);drawLines(mapKit,mapRouteBuf,mapRouteBuf.count,new Float32Array([0.15,0.23,0.32,0.35]),M,mapGl.LINE_STRIP);let [ctx,w,h]=fit2(mapOv);ctx.clearRect(0,0,w,h);drawRouteOverlay(ctx,w,h,cam,5);let p=project([s.x,s.y,s.z+2.2],cam,w,h);if(p){drawDroneOverlay(ctx,p,heading(s.i),34);ctx.fillStyle=riskColor(s.risk,D.bad[s.i]);ctx.beginPath();ctx.arc(p[0]+20,p[1]-17,6,0,7);ctx.fill()}}
function drawPlot(id,series,colors,ymin=null,ymax=null){let [ctx,w,h]=fit2($(id));ctx.clearRect(0,0,w,h);ctx.fillStyle="#ffffff";ctx.fillRect(0,0,w,h);let ml=44,mr=18,mt=34,mb=28,end=idxAt(simT),start=Math.max(0,end-900),ts=D.t.slice(start,end+1);if(!ts.length)return;let vals=series.flatMap(s=>s.slice(start,end+1).filter(Number.isFinite));let lo=ymin??Math.min(...vals),hi=ymax??Math.max(...vals);if(!isFinite(lo)||!isFinite(hi)){lo=0;hi=1}if(hi-lo<1e-6)hi=lo+1;ctx.strokeStyle="#dbe3ec";ctx.lineWidth=1;ctx.fillStyle="#64748b";ctx.font="11px Inter,system-ui";for(let i=0;i<=4;i++){let y=mt+(h-mt-mb)*i/4,val=hi-(hi-lo)*i/4;ctx.beginPath();ctx.moveTo(ml,y);ctx.lineTo(w-mr,y);ctx.stroke();ctx.fillText(val.toFixed(id==="risk"?1:0),6,y+4)}let px=i=>ml+(w-ml-mr)*(D.t[start+i]-D.t[start])/(D.t[end]-D.t[start]+1e-6),py=v=>mt+(hi-v)/(hi-lo)*(h-mt-mb);series.forEach((arr,si)=>{ctx.strokeStyle=colors[si];ctx.lineWidth=2;ctx.beginPath();let on=false;for(let i=0;i<ts.length;i++){let v=arr[start+i];if(!Number.isFinite(v)){on=false;continue}on?ctx.lineTo(px(i),py(v)):(ctx.moveTo(px(i),py(v)),on=true)}ctx.stroke()});ctx.strokeStyle="#94a3b8";ctx.beginPath();ctx.moveTo(ml,mt);ctx.lineTo(ml,h-mb);ctx.lineTo(w-mr,h-mb);ctx.stroke();ctx.fillStyle="#334155";ctx.fillText(`${D.t[start].toFixed(0)}s`,ml,h-8);ctx.fillText(`${D.t[end].toFixed(0)}s`,w-mr-34,h-8)}
function tick(now){let dt=(now-last)/1000;last=now;if(playing)simT=Math.min(maxT,simT+dt*speed);if(simT>=maxT)playing=false;let s=sample(simT),cam=drawWebGL(s);$("scrub").value=(simT-minT)/(maxT-minT);$("clock").textContent=simT.toFixed(1)+"s";let gnssBad=D.bad[s.i]?"BAD":"OK";$("hud").textContent=`ENU ${s.x.toFixed(1)}, ${s.y.toFixed(1)}, z=${s.z.toFixed(1)}m · risk=${s.risk.toFixed(2)} · GNSS ${gnssBad}`;$("mRisk").textContent=s.risk.toFixed(2);$("mAlt").textContent=s.z.toFixed(1)+"m";$("mVdop").textContent=(D.vdop[s.i]||0).toFixed(2);$("mRange").textContent=D.range[s.i]?"ON":"OFF";$("mGnssStatus").textContent=gnssBad;drawOverlay(s,cam);drawMapWebGL(s);drawPlot("plot",[D.z,D.gnss,D.ml],["#111827","#94a3b8","#0284c7"]);drawPlot("risk",[D.risk,D.vdop],["#7c3aed","#f97316"],0,Math.max(5,...D.vdop.filter(Number.isFinite)));requestAnimationFrame(tick)}
requestAnimationFrame(tick);
</script>
</body></html>
"""
