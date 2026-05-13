from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.world.scene import MeshScene


def make_planner_html(cfg: SimConfig, out_html: Path) -> None:
    scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    base = {
        "name": cfg.name,
        "seed": cfg.seed,
        "duration_s": cfg.duration_s,
        "dt_s": cfg.dt_s,
        "scene": asdict(cfg.scene),
        "mission": {
            "cruise_speed_mps": cfg.mission.cruise_speed_mps,
            "waypoint_acceptance_m": cfg.mission.waypoint_acceptance_m,
            "route_planner": cfg.mission.route_planner,
            "planner_grid_m": cfg.mission.planner_grid_m,
            "planner_clearance_m": cfg.mission.planner_clearance_m,
            "waypoints": [],
        },
        "sensors": asdict(cfg.sensors),
        "fusion": asdict(cfg.fusion),
        "energy": asdict(cfg.energy),
    }
    payload = {
        "baseConfig": base,
        "mesh": scene.webgl_mesh(),
        "bounds": scene.stats().__dict__,
        "existing": cfg.mission.waypoints,
    }
    html = _HTML.replace("__PAYLOAD__", json.dumps(payload))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GNSS-VIM WebGL Mission Studio</title>
<style>
  :root{--bg:#070b12;--panel:#0c1422;--panel2:#111c2e;--line:#26364f;--text:#eef6ff;--muted:#9aacc7;--accent:#38bdf8;--yellow:#fbbf24;--green:#22c55e;--red:#ef4444}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,sans-serif;overflow:hidden}
  .app{height:100vh;display:grid;grid-template-columns:1fr 390px}
  .stage{position:relative;overflow:hidden;background:linear-gradient(180deg,#111827,#07111f)}
  canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
  #overlay{pointer-events:none}
  .toolbar{position:absolute;left:16px;top:16px;display:flex;gap:8px;z-index:5}
  button{background:#2563eb;color:white;border:1px solid #3b82f6;border-radius:9px;padding:10px 12px;font-weight:750;cursor:pointer}
  button.secondary{background:#172033;border-color:#334155}
  button.danger{background:#b91c1c;border-color:#dc2626}
  button.good{background:#15803d;border-color:#22c55e}
  .hint{position:absolute;left:16px;bottom:16px;z-index:5;background:#07111fcc;border:1px solid var(--line);border-radius:12px;padding:12px 14px;color:var(--muted);font-size:13px;line-height:1.45;backdrop-filter:blur(8px)}
  aside{background:linear-gradient(180deg,#101827,#0a111d);border-left:1px solid var(--line);padding:18px;overflow:auto}
  h1{font-size:19px;margin:0 0 5px}
  .sub{color:var(--muted);font-size:13px;line-height:1.45;margin-bottom:14px}
  label{display:block;margin-top:11px;color:#cbd5e1;font-size:12px}
  input,select,textarea{width:100%;margin-top:5px;border-radius:8px;border:1px solid #334155;background:#07111f;color:var(--text);padding:9px}
  textarea{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;min-height:170px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
  .buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
  .card{border:1px solid var(--line);background:#08111f;border-radius:12px;padding:12px;margin-top:14px}
  .wp{display:grid;grid-template-columns:22px 1fr 60px;gap:8px;align-items:center;border:1px solid #203047;background:#0d1624;border-radius:9px;padding:8px;margin-top:7px;cursor:pointer}
  .wp.active{border-color:var(--accent);box-shadow:0 0 0 1px #38bdf8 inset}
  .dot{width:12px;height:12px;border-radius:50%;background:var(--yellow)}
  .wp b{font-size:13px}.wp span{display:block;color:var(--muted);font-size:11px;margin-top:2px}
  .pill{font-size:11px;color:#bfdbfe;text-align:right}
</style>
</head>
<body>
<div class="app">
  <main class="stage" id="stage">
    <canvas id="gl"></canvas>
    <canvas id="overlay"></canvas>
    <div class="toolbar">
      <button class="secondary" id="viewIso">Iso</button>
      <button class="secondary" id="viewTop">Top</button>
      <button class="secondary" id="viewReset">Reset</button>
      <button class="secondary" id="toggleGrid">Grid</button>
    </div>
    <div class="hint">Left-drag orbit. Right/shift-drag pan. Wheel zoom. Click ground to add a waypoint. Click a waypoint to select and edit it.</div>
  </main>
  <aside>
    <h1>GNSS-VIM Mission Studio</h1>
    <div class="sub">Self-contained WebGL planner rendering the actual OSM/Blender mesh. Export the mission JSON, then run it with <code>gnss-vim-sim run --config</code>.</div>
    <div class="row">
      <div><label>Point type</label><select id="mode"><option>takeoff</option><option selected>waypoint</option><option>landing</option></select></div>
      <div><label>Altitude ENU m</label><input id="alt" type="number" value="8" step="0.5"></div>
    </div>
    <label>Name</label><input id="name" value="wp">
    <div class="triple">
      <div><label>X east</label><input id="x" type="number" step="0.1"></div>
      <div><label>Y north</label><input id="y" type="number" step="0.1"></div>
      <div><label>Z up</label><input id="z" type="number" step="0.1"></div>
    </div>
    <div class="buttons">
      <button class="good" id="apply">Apply Selected</button>
      <button class="danger" id="delete">Delete Selected</button>
      <button class="secondary" id="load">Load Existing</button>
      <button class="secondary" id="clear">New Mission</button>
    </div>
    <button style="width:100%;margin-top:10px" id="download">Download Mission Config</button>
    <div class="card"><b>Waypoints</b><div id="list"></div></div>
    <label>Generated Config</label><textarea id="json" readonly></textarea>
  </aside>
</div>
<script>
const P=__PAYLOAD__;
const mesh=P.mesh||{positions:[],normals:[],indices:[],mode:"empty"};
let waypoints=[], selected=-1, showGrid=true;
const glCanvas=document.getElementById("gl"), ov=document.getElementById("overlay"), stage=document.getElementById("stage");
const gl=glCanvas.getContext("webgl",{antialias:true});
if(!gl){alert("WebGL is required for the mission planner");}
const $=id=>document.getElementById(id);
let bounds=P.bounds&&P.bounds.bounds_min?[P.bounds.bounds_min,P.bounds.bounds_max]:[[-250,-250,0],[250,250,40]];
let center=[(bounds[0][0]+bounds[1][0])/2,(bounds[0][1]+bounds[1][1])/2,0];
let sceneSpan=Math.max(bounds[1][0]-bounds[0][0],bounds[1][1]-bounds[0][1],120);
let camera={target:center.slice(),dist:sceneSpan*1.05,az:0.78,el:0.78,fov:50*Math.PI/180};
let mouse={down:false,x:0,y:0,button:0,moved:0};

function v3(x=0,y=0,z=0){return [x,y,z]}
function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function muls(a,s){return [a[0]*s,a[1]*s,a[2]*s]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function norm(a){let l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l]}
function eye(){let ce=Math.cos(camera.el);return [camera.target[0]+camera.dist*ce*Math.sin(camera.az),camera.target[1]+camera.dist*ce*Math.cos(camera.az),camera.target[2]+camera.dist*Math.sin(camera.el)]}
function basis(){let e=eye(),f=norm(sub(camera.target,e)),r=norm(cross(f,[0,0,1])),u=norm(cross(r,f));return {e,f,r,u}}
function perspective(fovy,aspect,near,far){let f=1/Math.tan(fovy/2),nf=1/(near-far);return [f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]}
function lookAt(e,c,up){let z=norm(sub(e,c)),x=norm(cross(up,z)),y=cross(z,x);return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,e),-dot(y,e),-dot(z,e),1]}
function matMul(a,b){let o=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
function project(p){let b=basis(),rel=sub(p,b.e),z=dot(rel,b.f);if(z<=0)return null;let rect=ov.getBoundingClientRect(),tan=Math.tan(camera.fov/2),asp=rect.width/rect.height;return [rect.width/2+(dot(rel,b.r)/(z*tan*asp))*rect.width/2,rect.height/2-(dot(rel,b.u)/(z*tan))*rect.height/2,z]}
function rayGround(sx,sy){let rect=ov.getBoundingClientRect(),b=basis(),asp=rect.width/rect.height,tan=Math.tan(camera.fov/2);let nx=(sx/rect.width*2-1)*asp*tan,ny=(1-sy/rect.height*2)*tan;let d=norm(add(add(b.f,muls(b.r,nx)),muls(b.u,ny)));let t=-b.e[2]/d[2];if(!isFinite(t)||t<=0)return null;return add(b.e,muls(d,t))}

function shader(type,src){let s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
function program(vs,fs){let p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p}
const meshProg=program(
`attribute vec3 aPos;attribute vec3 aNor;uniform mat4 uMvp;varying vec3 vNor;varying float vZ;void main(){vNor=aNor;vZ=aPos.z;gl_PointSize=2.5;gl_Position=uMvp*vec4(aPos,1.0);}`,
`precision mediump float;varying vec3 vNor;varying float vZ;void main(){vec3 n=normalize(vNor+vec3(0.0,0.0,0.001));float l=max(dot(n,normalize(vec3(-0.45,-0.35,0.82))),0.0);vec3 low=vec3(0.18,0.25,0.33);vec3 high=vec3(0.72,0.78,0.84);vec3 c=mix(low,high,clamp(vZ/32.0,0.0,1.0))*(0.42+0.72*l);gl_FragColor=vec4(c,1.0);}`
);
const lineProg=program(
`attribute vec3 aPos;uniform mat4 uMvp;void main(){gl_Position=uMvp*vec4(aPos,1.0);}`,
`precision mediump float;uniform vec4 uColor;void main(){gl_FragColor=uColor;}`
);
function buf(data,itemSize){let b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);return {b,itemSize,count:data.length/itemSize}}
const posBuf=buf(mesh.positions,3);
const norBuf=buf(mesh.normals.length?mesh.normals:new Array(mesh.positions.length).fill(0).map((_,i)=>i%3===2?1:0),3);
let idxBuf=null, idxCount=0, idxType=gl.UNSIGNED_SHORT;
if(mesh.indices.length){let vertexCount=mesh.positions.length/3, use32=vertexCount>65535;if(use32&&!gl.getExtension("OES_element_index_uint"))throw new Error("Mesh needs 32-bit WebGL indices, but this browser does not expose OES_element_index_uint");idxBuf=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,use32?new Uint32Array(mesh.indices):new Uint16Array(mesh.indices),gl.STATIC_DRAW);idxCount=mesh.indices.length;idxType=use32?gl.UNSIGNED_INT:gl.UNSIGNED_SHORT}
function gridData(){let d=[],step=20,x0=Math.floor((bounds[0][0]-60)/step)*step,x1=Math.ceil((bounds[1][0]+60)/step)*step,y0=Math.floor((bounds[0][1]-60)/step)*step,y1=Math.ceil((bounds[1][1]+60)/step)*step;for(let x=x0;x<=x1;x+=step){d.push(x,y0,0,x,y1,0)}for(let y=y0;y<=y1;y+=step){d.push(x0,y,0,x1,y,0)}return d}
const gridBuf=buf(gridData(),3);

function resize(){let d=window.devicePixelRatio||1,r=stage.getBoundingClientRect();for(const c of [glCanvas,ov]){if(c.width!==r.width*d||c.height!==r.height*d){c.width=r.width*d;c.height=r.height*d;c.style.width=r.width+"px";c.style.height=r.height+"px"}}gl.viewport(0,0,glCanvas.width,glCanvas.height)}
function mvp(){let e=eye(),view=lookAt(e,camera.target,[0,0,1]),proj=perspective(camera.fov,glCanvas.width/glCanvas.height,0.2,Math.max(5000,camera.dist*8));return matMul(proj,view)}
function drawGL(){resize();gl.clearColor(0.035,0.055,0.09,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);let M=new Float32Array(mvp());
 if(showGrid){gl.useProgram(lineProg);let a=gl.getAttribLocation(lineProg,"aPos");gl.bindBuffer(gl.ARRAY_BUFFER,gridBuf.b);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,"uMvp"),false,M);gl.uniform4f(gl.getUniformLocation(lineProg,"uColor"),0.18,0.28,0.42,0.55);gl.drawArrays(gl.LINES,0,gridBuf.count)}
 gl.useProgram(meshProg);let ap=gl.getAttribLocation(meshProg,"aPos"),an=gl.getAttribLocation(meshProg,"aNor");gl.bindBuffer(gl.ARRAY_BUFFER,posBuf.b);gl.enableVertexAttribArray(ap);gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,norBuf.b);gl.enableVertexAttribArray(an);gl.vertexAttribPointer(an,3,gl.FLOAT,false,0,0);gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,"uMvp"),false,M);if(idxBuf){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf);gl.drawElements(gl.TRIANGLES,idxCount,idxType,0)}else{gl.drawArrays(gl.POINTS,0,posBuf.count)}
}
function drawOverlay(){let ctx=ov.getContext("2d"),d=window.devicePixelRatio||1,w=ov.width/d,h=ov.height/d;ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);
 ctx.lineWidth=3;ctx.strokeStyle="#38bdf8";ctx.beginPath();let started=false;for(const wp of waypoints){let p=project([wp.x,wp.y,wp.z]);if(!p)continue;if(!started){ctx.moveTo(p[0],p[1]);started=true}else ctx.lineTo(p[0],p[1])}ctx.stroke();
 for(let i=0;i<waypoints.length;i++){let wp=waypoints[i],p=project([wp.x,wp.y,wp.z]);if(!p)continue;ctx.fillStyle=i===selected?"#22c55e":"#fbbf24";ctx.strokeStyle="#07111f";ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],8,0,Math.PI*2);ctx.fill();ctx.stroke();if(i===selected){ctx.strokeStyle="#38bdf8";ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],15,0,Math.PI*2);ctx.stroke()}ctx.fillStyle="#eaf2ff";ctx.font="12px Inter,system-ui";ctx.fillText(wp.name,p[0]+12,p[1]-10)}
 ctx.fillStyle="#dbeafe";ctx.font="13px Inter,system-ui";ctx.fillText(`mesh mode: ${mesh.mode} | vertices: ${Math.floor(mesh.positions.length/3)} | faces: ${Math.floor(mesh.indices.length/3)}`,16,54);
}
function frame(){drawGL();drawOverlay();requestAnimationFrame(frame)}

function wpName(mode){if(mode==="takeoff")return "takeoff";if(mode==="landing")return "land";let n=$("name").value.trim();return n&&n!=="wp"?n:`wp_${waypoints.length+1}`}
function addWaypoint(p){let mode=$("mode").value,z=parseFloat($("alt").value||"8");waypoints.push({name:wpName(mode),x:+p[0].toFixed(3),y:+p[1].toFixed(3),z:+z.toFixed(3)});selected=waypoints.length-1;if(mode==="takeoff")$("mode").value="waypoint";syncFields();refreshList()}
function syncFields(){let wp=waypoints[selected];if(!wp)return;$("name").value=wp.name;$("alt").value=wp.z;$("x").value=wp.x.toFixed(2);$("y").value=wp.y.toFixed(2);$("z").value=wp.z.toFixed(2)}
function applySelected(){if(selected<0||!waypoints[selected])return;let wp=waypoints[selected];wp.name=$("name").value.trim()||wp.name;wp.x=parseFloat($("x").value||wp.x);wp.y=parseFloat($("y").value||wp.y);wp.z=parseFloat($("z").value||$("alt").value||wp.z);$("alt").value=wp.z;refreshList()}
function deleteSelected(){if(selected<0)return;waypoints.splice(selected,1);selected=Math.min(selected,waypoints.length-1);syncFields();refreshList()}
function refreshList(){let list=$("list");list.innerHTML=waypoints.map((w,i)=>`<div class="wp ${i===selected?"active":""}" onclick="selectWp(${i})"><div class="dot"></div><div><b>${i+1}. ${w.name}</b><span>x=${w.x.toFixed(1)} y=${w.y.toFixed(1)} z=${w.z.toFixed(1)}</span></div><div class="pill">${i===0?"start":i===waypoints.length-1?"end":"wp"}</div></div>`).join("");let cfg=structuredClone(P.baseConfig);cfg.mission.waypoints=waypoints.map(w=>({name:w.name,x:+w.x.toFixed(3),y:+w.y.toFixed(3),z:+w.z.toFixed(3)}));$("json").value=JSON.stringify(cfg,null,2)}
window.selectWp=i=>{selected=i;syncFields();refreshList()}
function loadExisting(){waypoints=(P.existing||[]).map(w=>({name:w.name,x:+w.x,y:+w.y,z:+w.z}));selected=waypoints.length?0:-1;syncFields();refreshList()}
function downloadConfig(){let blob=new Blob([$("json").value],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="mission_config.json";a.click();URL.revokeObjectURL(a.href)}
$("apply").onclick=applySelected;$("delete").onclick=deleteSelected;$("load").onclick=loadExisting;$("clear").onclick=()=>{waypoints=[];selected=-1;refreshList()};$("download").onclick=downloadConfig;
$("viewIso").onclick=()=>{camera.az=.78;camera.el=.78};$("viewTop").onclick=()=>{camera.az=0;camera.el=Math.PI/2-0.01};$("viewReset").onclick=()=>{camera.target=center.slice();camera.dist=sceneSpan*1.05;camera.az=.78;camera.el=.78};$("toggleGrid").onclick=()=>{showGrid=!showGrid};
for(const id of ["name","x","y","z","alt"])$(id).addEventListener("change",()=>{if(id==="alt"&&selected>=0){$("z").value=$("alt").value}applySelected()});
stage.addEventListener("contextmenu",e=>e.preventDefault());
stage.addEventListener("pointerdown",e=>{mouse={down:true,x:e.clientX,y:e.clientY,button:e.button,moved:0};stage.setPointerCapture(e.pointerId)});
stage.addEventListener("pointermove",e=>{if(!mouse.down)return;let dx=e.clientX-mouse.x,dy=e.clientY-mouse.y;mouse.x=e.clientX;mouse.y=e.clientY;mouse.moved+=Math.abs(dx)+Math.abs(dy);let b=basis();if(mouse.button===2||e.shiftKey){let s=camera.dist*0.0018;camera.target=add(camera.target,add(muls(b.r,-dx*s),muls([b.f[0],b.f[1],0],dy*s)))}else{camera.az-=dx*0.006;camera.el=Math.max(0.08,Math.min(Math.PI/2-0.02,camera.el+dy*0.006))}});
stage.addEventListener("pointerup",e=>{if(mouse.moved<5&&e.button===0){let r=ov.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;let hit=-1,best=18;for(let i=0;i<waypoints.length;i++){let p=project([waypoints[i].x,waypoints[i].y,waypoints[i].z]);if(!p)continue;let d=Math.hypot(p[0]-sx,p[1]-sy);if(d<best){best=d;hit=i}}if(hit>=0){selected=hit;syncFields();refreshList()}else{let g=rayGround(sx,sy);if(g)addWaypoint(g)}}mouse.down=false});
stage.addEventListener("wheel",e=>{e.preventDefault();camera.dist*=Math.exp(e.deltaY*0.001);camera.dist=Math.max(20,Math.min(sceneSpan*5,camera.dist))},{passive:false});
loadExisting();requestAnimationFrame(frame);
</script>
</body>
</html>
"""
