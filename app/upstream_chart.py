"""公式/理工科图表上游：让 gpt-5.5 输出 **HTML 正文片段**（支持 MathJax 公式与 Mermaid
流程图），服务器套进固定模板存成自包含 HTML，由浏览器端渲染——排版远胜 matplotlib，
且天然支持流程图。前端用 iframe 显示、html2canvas 导出 PNG、或直接下载 HTML。

兼容两种 OpenAI 文本接口：Chat Completions(/chat/completions) 与 Responses(/responses)。
由管理面板 provider="chart" 的 api_style 决定（auto/chat/responses）：auto 会按模型名
自动判别——gpt-5.x 与 o 系列推理模型走 Responses API，其余走 Chat Completions。

配置（base/key/model/api_style）取自管理面板 provider="chart"，不依赖环境变量。
"""
from __future__ import annotations

import logging
import re

import httpx

from . import db

log = logging.getLogger("image2.chart")

# 已异步化（后台任务 + 轮询），不受 Cloudflare 100s 限制，读超时给足
_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0)
_HTTP_RETRIES = 1


class ChartError(RuntimeError):
    pass


_SYSTEM_PROMPT = """你是科学排版与信息图专家。根据用户需求，输出一段 **HTML 正文片段**
（只输出 <body> 内部的内容），由浏览器渲染成排版精良的图文。

# 能力与写法（按需选型，优先用专业库，别手画 SVG）
- 数据图表（折线/柱状/散点/饼图/环形/雷达/K线/箱线/热力图/桑基/仪表盘等）：**首选 ECharts**。
  写成 <div class="echarts" data-h="360">{ ECharts option 的 JSON }</div>，div 的文本就是一个
  ECharts `option` 的**合法 JSON**：双引号、无尾逗号、无注释、无函数、不要出现裸 < 或 >。
  配色 / 字体 / 留白已有统一主题，你只需给数据与必要的 title / tooltip / 坐标轴 / series。例如：
  <div class="echarts" data-h="340">{"title":{"text":"季度销量"},"tooltip":{"trigger":"axis"},"xAxis":{"type":"category","data":["Q1","Q2","Q3","Q4"]},"yAxis":{"type":"value"},"series":[{"type":"bar","name":"销量","data":[120,200,150,260]}]}</div>
  多个数据系列就在 series 里加多项；想要折线就 "type":"line"，饼图 "type":"pie" 并用 {"name","value"}。
- 关键指标卡（KPI / 概览大数字，适合放开头做摘要）：用
  <div class="metrics">[{"label":"营收","value":"¥1.28M","delta":"+12.4%","trend":"up"}, ...]</div>，
  div 文本是 JSON 数组，每项 {label, value, delta?, trend?}（trend 取 up/down/flat）。
- 数学公式：用 MathJax。行内 \\( ... \\) 或 $...$，独立成行 $$ ... $$ 或 \\[ ... \\]。
- 架构图 / 框图 / 管线 pipeline / 数据流 / 模块关系 等**节点-连线类图：首选结构化 diagram 块**。
  你只给“图的语义”（哪些节点、谁连谁），由模板自动排版、自动按文字撑开方框、自动配色——
  **永不截断、永不黑底黑字、导出 PNG 也稳**。写成 <div class="diagram" data-dir="LR"> { JSON } </div>，
  data-dir 取 LR(横向,默认) / TB(纵向) / RL / BT。JSON 形如：
  <div class="diagram" data-dir="LR">{"nodes":[{"id":"a","label":"力历史 20 步","kind":"input"},{"id":"b","label":"归一化 → 投影","kind":"process","group":"接触编码器"},{"id":"c","label":"几何时序 token","kind":"data","group":"接触编码器"},{"id":"d","label":"VLA 动作专家","kind":"model"},{"id":"e","label":"机器人动作","kind":"output"}],"edges":[{"from":"a","to":"b","label":"force"},{"from":"b","to":"c"},{"from":"c","to":"d","label":"token"},{"from":"d","to":"e"}]}</div>
  · node：id（唯一，只用英文/数字，给边引用）、label（显示文字，可中文，写完整词、随便多长都不会截断）、
    kind?（决定品牌配色：input 输入 / process 处理 / model 模型 / output 输出 / data 数据 /
    store 存储 / decision 判断 / external 外部）、
    shape?（rect 方框默认 / round 圆角 / stadium 胶囊 / cylinder 圆柱 / diamond 菱形判断 / hexagon 六边）、
    group?（同名的若干节点会被框进一个带标题的子框 subgraph，用来表达“某模块内含几步”）。
  · edge：from / to（写对应 node 的 id）、label?（连线上的文字）。
  · 你**绝不要**写任何坐标、宽高、颜色、SVG、classDef——只描述节点与连线，其余全交给模板。
  · 颜色靠 kind 自动上品牌色，不用画图例(legend)；要分组就用 group。
- 时序图 / 状态机 / 类图 / ER 图（diagram 块表达不了的）仍可直接写 <pre class="mermaid"> ... </pre>，
  横向 graph LR、纵向 graph TD；节点文字写在 [] 里，Mermaid 会按文字自动撑开，不会截断。
- **严禁手画 SVG 复杂图 / 场景 / 装置 / 插画**：框图、流程图、机器人 / 机械臂 / 设备 / 3D /
  物理场景示意、带多个部件或多个标注的"配图"——LLM 手画必然**松散、错位、部件断开、互相重叠、
  比例失调，非常难看**（你之前画的机械臂+曲面就是反面教材）。这类需求**一律改用 diagram 结构化块**：
  把"机器人 → 末端工具 → 曲面 / 接触点 / 法向力"这种关系画成节点-连线图（用 node 表示部件/概念、
  用 edge 表示作用关系、用 label 标注力/接触等），或干脆用**文字 + 表格 + 公式**讲清楚，再配
  ECharts 数据图。**不要试图"画出"一个物理画面**。
- 内联 <svg> 只允许**极简、单一、无场景**的小示意：一个小图标、一条带刻度的坐标轴、一个单独的
  几何形（如一个标注了边长的三角形）。即便如此也要：每个形状显式写 fill/stroke（默认黑色！）、
  浅底深字、根 <svg> 用 width="100%"+盖全 viewBox、文字 text-anchor="middle" 留足框宽。
  **凡是需要多个部件拼出一幅画的，全部走 diagram 块或文字，绝不手画。**
- 文字排版：用语义化标签 h1/h2/h3/p/ul/ol/table/figure/figcaption/blockquote/code 等。
- 用一个 <h1> 作为整篇标题；需要时给表格加 <caption>。可配合小标题、要点列表、数据表把内容讲清楚。

# 选型建议（强约束）
- 数值/趋势/占比/分布 → ECharts。**节点-连线/架构/框图/流程/管线/物理关系示意 → diagram 结构化块
  （严禁手画 SVG 场景/装置！）**；时序/状态/类/ER → mermaid 块。公式 → MathJax。
  手画 SVG 只剩"极简单一示意"（单图标/坐标轴/单几何形）这一条窄路，复杂的一律走结构化块或文字。

# 硬性约定
- 只输出 <body> 内部的 HTML 片段；不要写 <html>/<head>/<body>/<script>/<link>。
  （<style> 可以用，但仅用于给你自己的图/SVG 着色，作用域到自定义 class，别覆盖全局标签。）
  ECharts / MathJax / Mermaid / diagram 都由外层模板统一提供，你只负责内容与结构。
- ECharts 的 option 必须是能被 JSON.parse 的纯 JSON（不能写 JS 表达式 / 函数 / 变量）。
- diagram 块的文本必须是能被 JSON.parse 的纯 JSON（双引号、无尾逗号、无注释、无函数、不要裸 < >）；
  node 的 id 只用英文/数字，label 才放中文；edge 的 from/to 必须能对上某个 node 的 id。
- 内联 SVG 的每个 rect/circle/path/text 都要显式 fill 与 stroke，绝不依赖默认值。
- 不要引用任何外部图片/字体/JS；不要写 onclick 等事件属性。
- 正文用中文；公式、变量、代码保持原样。

# 输出
直接输出 HTML 片段本身（可放在 ```html 代码块里），不要任何额外解释。
"""


def _repair_prompt() -> str:
    return ('请只输出 <body> 内的 HTML 正文片段：数据图表用 '
            '<div class="echarts" data-h="360">{合法的 ECharts option JSON}</div>；'
            '架构/流程/框图/管线/节点关系图用 <div class="diagram" data-dir="LR">'
            '{nodes/edges 的纯 JSON}</div>（模板会自动排版、按文字撑开、不截断）——'
            '严禁手画 SVG 框图；若上一版用 SVG 画了方块流程图，请改写成等价的 diagram 块。'
            '时序/状态/类/ER 图可用 <pre class="mermaid">。公式用 MathJax。'
            '严禁手画 SVG 场景/装置/机械臂/物理配图（必然松散错位）——这类改成 diagram 块或文字+表格；'
            '内联 SVG 只留给极简单一示意（单图标/坐标轴/单个几何形），且形状显式写 fill/stroke。'
            '不要 <html>/<head>/<script>，不要解释。')


def extract_html(text: str) -> str:
    """从模型回复里取出 HTML 正文片段。

    保留 <style> 与内联样式——模型常用自定义 SVG/HTML 画图并用 <style>/class 着色，
    一旦剥掉，SVG 形状会回退成默认黑色填充、文字黑底黑字（就是那种“一堆黑方块”）。
    只移除真正危险的东西：<script>、<link>、on* 事件属性，以及 CSS 里的 @import /
    expression() / javascript:（沙箱 iframe 内本就隔离，风险极小）。
    """
    if not text:
        raise ChartError("返回内容为空")
    m = re.search(r"```(?:html?)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    frag = (m.group(1) if m else text).strip()
    bm = re.search(r"<body[^>]*>(.*)</body>", frag, flags=re.DOTALL | re.IGNORECASE)
    if bm:
        frag = bm.group(1).strip()
    frag = re.sub(r"</?(?:html|head|body)[^>]*>", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"<script\b.*?</script>", "", frag, flags=re.DOTALL | re.IGNORECASE)
    frag = re.sub(r"<link\b[^>]*>", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", frag, flags=re.IGNORECASE)  # 去事件属性
    frag = re.sub(r"@import\b[^;]*;", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"expression\s*\(", "_off_(", frag, flags=re.IGNORECASE)
    frag = re.sub(r"javascript\s*:", "", frag, flags=re.IGNORECASE)
    if "<" not in frag or len(frag) < 10:
        raise ChartError(f"未获得有效 HTML 内容。返回片段：{text[:300]}")
    return frag


def wrap_html(body_fragment: str) -> str:
    """把 LLM 的正文片段套进自包含模板（含 MathJax/Mermaid/html2canvas 与快照监听）。"""
    return _HTML_TEMPLATE.replace("__BODY__", body_fragment)


# 自包含 HTML 模板：浏览器端渲染。__BODY__ 处填入 LLM 的正文片段。
# 监听 parent 的 {type:'snapshot'} 消息 → 等公式/流程图渲染完 → html2canvas 导出 PNG 回传。
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{color-scheme:light}
  *{box-sizing:border-box}
  body{margin:0;background:#fff;color:#1A1A17;
    font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    line-height:1.7;font-size:15px;padding:32px 44px;max-width:940px;margin:0 auto;
    -webkit-font-smoothing:antialiased;}
  h1{font-size:24px;font-weight:700;letter-spacing:-.02em;margin:0 0 16px}
  h2{font-size:19px;font-weight:650;margin:28px 0 10px}
  h3{font-size:16px;font-weight:650;margin:22px 0 8px}
  p{margin:10px 0}
  ul,ol{margin:10px 0;padding-left:22px}
  li{margin:4px 0}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
  th,td{border:1px solid #E7E6E0;padding:7px 10px;text-align:left;vertical-align:top}
  th{background:#FAFAF8;font-weight:600}
  caption{caption-side:top;text-align:left;color:#8C8B81;font-size:12.5px;margin-bottom:6px}
  code{font-family:"JetBrains Mono",Consolas,monospace;background:#F4F4F1;padding:1px 5px;border-radius:4px;font-size:13px}
  pre{background:#F7F7F4;border:1px solid #EEEDE8;border-radius:8px;padding:12px 14px;overflow:auto}
  pre.mermaid{background:transparent;border:0;padding:0;text-align:center;overflow:visible}
  blockquote{margin:12px 0;padding:6px 14px;border-left:3px solid #F0CDBC;color:#57564E;background:#FBEDE6;border-radius:0 6px 6px 0}
  figure{margin:16px 0;text-align:center}
  figcaption{color:#8C8B81;font-size:12.5px;margin-top:6px}
  svg{max-width:100%;height:auto}
  a{color:#D9531E}
  hr{border:0;border-top:1px solid #E7E6E0;margin:20px 0}
  mjx-container[display]{margin:14px 0!important}
  .echarts{width:100%;margin:16px auto;}
  .diagram{display:none;}
  .mermaid{width:100%;margin:16px 0;text-align:center;}
  .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0;}
  .metric-card{border:1px solid #E7E6E0;border-radius:12px;padding:14px 16px;background:#fff;}
  .metric-card .ml{font-size:12px;color:#8C8B81;}
  .metric-card .mv{font-size:24px;font-weight:700;color:#1A1A17;letter-spacing:-.02em;margin-top:4px;line-height:1.15;}
  .metric-card .md{font-size:12px;margin-top:5px;font-weight:600;}
  .metric-card .md.up{color:#0E9F6E;} .metric-card .md.down{color:#C5403E;} .metric-card .md.flat{color:#8C8B81;}
</style>
<script>
  window.MathJax={tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']]},
    svg:{fontCache:'none'},options:{enableMenu:false}};
</script>
<script src="/static/vendor/echarts.min.js?v=5.5.1"></script>
<script async src="https://fastly.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script src="https://fastly.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script src="https://fastly.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
</head>
<body>
__BODY__
<script>
  function _waitFor(cond,ms){return new Promise(function(res){
    var t=setInterval(function(){if(cond()){clearInterval(t);res();}},50);
    setTimeout(function(){clearInterval(t);res();},ms||9000);});}
  // ECharts 统一主题：克制焦橙配色 + Inter 字体 + 轻网格，保证默认就好看
  var ECHARTS_THEME={
    color:['#D9531E','#2F6BD6','#0E9F6E','#B7791F','#7C3AED','#C5403E','#0891B2','#DB2777','#65A30D'],
    backgroundColor:'transparent',
    textStyle:{fontFamily:'Inter,"PingFang SC","Microsoft YaHei",sans-serif',color:'#57564E'},
    title:{textStyle:{color:'#1A1A17',fontSize:15,fontWeight:600},left:'center',top:6},
    legend:{textStyle:{color:'#57564E'},top:30},
    grid:{left:'5%',right:'5%',bottom:'7%',top:64,containLabel:true},
    categoryAxis:{axisLine:{lineStyle:{color:'#D8D7D0'}},axisTick:{show:false},axisLabel:{color:'#57564E'},splitLine:{show:false}},
    valueAxis:{axisLine:{show:false},axisTick:{show:false},axisLabel:{color:'#8C8B81'},splitLine:{lineStyle:{color:'#EEEDE8'}}},
    line:{smooth:true,symbolSize:6,lineStyle:{width:2.5}},
    bar:{itemStyle:{borderRadius:[4,4,0,0]}},
    pie:{itemStyle:{borderColor:'#fff',borderWidth:2}}
  };
  function _initEcharts(){
    if(!window.echarts)return;
    try{echarts.registerTheme('img2',ECHARTS_THEME);}catch(e){}
    var hosts=[];
    document.querySelectorAll('.echarts').forEach(function(el){
      var raw=(el.textContent||'').trim(); if(!raw)return;
      var opt; try{opt=JSON.parse(raw);}catch(e){el.textContent='图表配置 JSON 解析失败: '+e.message;el.style.color='#C5403E';return;}
      var h=parseInt(el.getAttribute('data-h'))||360;
      el.textContent='';el.style.height=h+'px';el.style.width='100%';
      try{var c=echarts.init(el,'img2',{renderer:'svg'});c.setOption(opt);hosts.push(c);}
      catch(e){el.textContent='图表渲染失败: '+e.message;el.style.color='#C5403E';}
    });
    if(hosts.length){window.addEventListener('resize',function(){hosts.forEach(function(c){try{c.resize();}catch(e){}});});}
  }
  function _initMetrics(){
    document.querySelectorAll('.metrics').forEach(function(el){
      var raw=(el.textContent||'').trim(); if(!raw)return;
      var arr; try{arr=JSON.parse(raw);}catch(e){el.textContent='指标卡 JSON 解析失败: '+e.message;el.style.color='#C5403E';return;}
      if(!Array.isArray(arr))return;
      el.textContent='';
      arr.forEach(function(m){
        var c=document.createElement('div'); c.className='metric-card';
        var lbl=document.createElement('div'); lbl.className='ml'; lbl.textContent=m.label||''; c.appendChild(lbl);
        var val=document.createElement('div'); val.className='mv'; val.textContent=(m.value==null?'':String(m.value)); c.appendChild(val);
        if(m.delta!=null&&m.delta!==''){var d=document.createElement('div'); d.className='md '+(m.trend==='up'?'up':m.trend==='down'?'down':'flat');
          d.textContent=(m.trend==='up'?'▲ ':m.trend==='down'?'▼ ':'')+String(m.delta); c.appendChild(d);}
        el.appendChild(c);
      });
    });
  }
  // 结构化 diagram JSON → Mermaid 源码：交给 Mermaid 自动排版/撑框/上品牌色，永不截断
  var DIAG_KIND2CLASS={input:'info',data:'data',process:'info',model:'model',
    output:'ok',store:'data',decision:'accent',external:'muted',accent:'accent'};
  var DIAG_SHAPE={rect:['[',']'],round:['(',')'],stadium:['([','])'],
    subroutine:['[[',']]'],cylinder:['[(',')]'],diamond:['{','}'],
    hexagon:['{{','}}'],parallelogram:['[/','/]']};
  function _diagId(s){return String(s==null?'':s).replace(/[^A-Za-z0-9_]/g,'_').slice(0,40)||'n';}
  function _diagLabel(s){return '"'+String(s==null?'':s).replace(/"/g,'&quot;').replace(/[\r\n]+/g,' ').replace(/ /g,' ').trim()+'"';}
  function _diagramToMermaid(spec,dir){
    var nodes=(spec&&spec.nodes)||[], edges=(spec&&spec.edges)||[];
    var d=/^(LR|RL|TB|BT)$/.test(dir||'')?dir:'LR';
    var out=['graph '+d], groups={}, seen={}, classLines=[];
    nodes.forEach(function(n){
      var id=_diagId(n.id); if(seen[id])return; seen[id]=1;
      var shape=n.shape||(n.kind==='decision'?'diamond':'rect');
      var w=DIAG_SHAPE[shape]||DIAG_SHAPE.rect;
      var line='  '+id+w[0]+_diagLabel(n.label)+w[1];
      if(n.group){(groups[n.group]=groups[n.group]||[]).push(line);}else{out.push(line);}
      var cls=DIAG_KIND2CLASS[n.kind]; if(cls)classLines.push('  class '+id+' '+cls);
    });
    Object.keys(groups).forEach(function(g,i){
      out.push('  subgraph sg'+i+'['+_diagLabel(g).slice(1,-1)+']');
      groups[g].forEach(function(l){out.push('  '+l);});
      out.push('  end');
    });
    edges.forEach(function(e){
      var a=_diagId(e.from), b=_diagId(e.to);
      if(!seen[a]||!seen[b])return;   // 丢弃指向不存在节点的边，避免凭空多出节点
      out.push('  '+a+(e.label?(' -->|'+_diagLabel(e.label)+'| '):' --> ')+b);
    });
    return out.concat(classLines).join('\n');
  }
  function _initDiagrams(){
    document.querySelectorAll('.diagram').forEach(function(el){
      var raw=(el.textContent||'').trim(); if(!raw)return;
      var spec; try{spec=JSON.parse(raw);}
      catch(e){var err=document.createElement('div');err.style.color='#C5403E';err.textContent='结构图 JSON 解析失败: '+e.message;el.replaceWith(err);return;}
      var pre=document.createElement('pre'); pre.className='mermaid';
      pre.textContent=_diagramToMermaid(spec, el.getAttribute('data-dir'));
      el.replaceWith(pre);
    });
  }
  window.__renderReady=(async function(){
    try{_initMetrics();}catch(e){}
    await _waitFor(function(){return window.echarts;},9000);
    try{_initEcharts();}catch(e){}
    await _waitFor(function(){return window.mermaid;},9000);
    try{_initDiagrams();}catch(e){}
    try{
      mermaid.initialize({
        startOnLoad:false, securityLevel:'strict', htmlLabels:false, theme:'base',
        themeVariables:{
          fontFamily:'Inter,"PingFang SC","Microsoft YaHei",sans-serif', fontSize:'14px',
          primaryColor:'#F7F5F2', primaryBorderColor:'#D8D7D0', primaryTextColor:'#1A1A17',
          lineColor:'#9A9A90', textColor:'#57564E',
          clusterBkg:'#FBFAF8', clusterBorder:'#E7E6E0',
          secondaryColor:'#EEF3FB', tertiaryColor:'#F0FAF5', edgeLabelBackground:'#FFFFFF'
        },
        flowchart:{htmlLabels:false, curve:'basis', nodeSpacing:46, rankSpacing:64, padding:14, useMaxWidth:true}
      });
      if(!document.getElementById('brandMermaidCSS')){
        var _st=document.createElement('style'); _st.id='brandMermaidCSS';
        _st.textContent=
          '.mermaid .node.accent rect,.mermaid .node.accent polygon,.mermaid .node.accent path{fill:#FBEDE6 !important;stroke:#D9531E !important;stroke-width:1.4px !important}'+
          '.mermaid .node.info rect,.mermaid .node.info polygon,.mermaid .node.info path{fill:#EEF3FB !important;stroke:#2F6BD6 !important;stroke-width:1.2px !important}'+
          '.mermaid .node.ok rect,.mermaid .node.ok polygon,.mermaid .node.ok path{fill:#E9F7F1 !important;stroke:#0E9F6E !important;stroke-width:1.2px !important}'+
          '.mermaid .node.model rect,.mermaid .node.model polygon,.mermaid .node.model path{fill:#F0EAFB !important;stroke:#7C3AED !important;stroke-width:1.2px !important}'+
          '.mermaid .node.data rect,.mermaid .node.data polygon,.mermaid .node.data path{fill:#FBF1DA !important;stroke:#B7791F !important;stroke-width:1.2px !important}'+
          '.mermaid .node.muted rect,.mermaid .node.muted polygon,.mermaid .node.muted path{fill:#F4F4F1 !important;stroke:#C9C8C1 !important;stroke-width:1px !important}'+
          '.mermaid .node .label,.mermaid .node text,.mermaid .node tspan{fill:#1A1A17 !important;color:#1A1A17 !important}';
        document.head.appendChild(_st);
      }
      await mermaid.run();
    }catch(e){}
    await _waitFor(function(){return window.MathJax&&MathJax.startup&&MathJax.startup.promise;},9000);
    try{await MathJax.startup.promise;}catch(e){}
    try{if(window.MathJax&&MathJax.typesetPromise)await MathJax.typesetPromise();}catch(e){}
  })();
  // 把文档真实高度上报给父窗口：iframe 是无 same-origin 的沙箱，父窗口读不到内部高度，
  // 只能由文档自己量好 scrollHeight 回传，父窗口据此让「报告卡」按内容高度自适应——
  // 不再卡在固定高度里内部滚动 / 裁切图表，短内容也不撑出大片空白。
  // 只量 body 的「内容高度」：body 高度由内容决定（auto），不含视口。
  // 切忌用 documentElement.scrollHeight——它至少等于视口（=父窗口刚设的 iframe 高度），
  // 会和「按上报值设 iframe 高度」形成正反馈，高度无界增长。
  function _docHeight(){var b=document.body;return Math.max(b.scrollHeight,b.offsetHeight);}
  var _lastH=0;
  function _reportHeight(){
    var h=_docHeight();
    if(Math.abs(h-_lastH)<2)return;   // 没有实质变化就不上报，避免抖动
    _lastH=h;
    try{parent.postMessage({type:'doc-height',height:h},'*');}catch(e){}
  }
  window.__renderReady.then(function(){
    _reportHeight();
    // 渲染完布局可能还在回流（字体/公式/Mermaid 异步撑开），多打几拍兜底
    [120,400,900,1800].forEach(function(ms){setTimeout(_reportHeight,ms);});
  });
  // 只观察 body 内容尺寸变化（合法回流）；不监听 window resize——
  // 父窗口改 iframe 高度会触发 iframe 内的 resize，那正是反馈环的源头。
  if(window.ResizeObserver){try{new ResizeObserver(_reportHeight).observe(document.body);}catch(e){}}
  window.addEventListener('message',async function(e){
    if(!e.data||e.data.type!=='snapshot')return;
    try{
      await window.__renderReady;
      await new Promise(function(r){setTimeout(r,150);});
      var canvas=await html2canvas(document.body,{backgroundColor:'#ffffff',scale:2,useCORS:true,
        windowWidth:document.body.scrollWidth,windowHeight:document.body.scrollHeight});
      parent.postMessage({type:'snapshot-result',dataUrl:canvas.toDataURL('image/png')},'*');
    }catch(err){parent.postMessage({type:'snapshot-error',message:String(err)},'*');}
  });
</script>
</body>
</html>
"""


def _is_responses_model(model: str) -> bool:
    """按模型名判断是否应走 OpenAI Responses API。gpt-5.x 与 o 系列推理模型默认走 Responses。"""
    m = (model or "").lower().strip()
    if "responses" in m:
        return True
    if m.startswith(("gpt-5", "gpt5")):
        return True
    if re.match(r"^o[1-9]", m):  # o1 / o3 / o4-mini 等
        return True
    return False


def _resolve_style(cfg: dict) -> str:
    style = (cfg.get("api_style") or "auto").lower()
    if style in ("chat", "responses"):
        return style
    return "responses" if _is_responses_model(cfg.get("model", "")) else "chat"


def _extract_responses_text(data: dict) -> str:
    """解析 OpenAI Responses API 返回的文本（兼容多种形状）。"""
    ot = data.get("output_text")  # SDK 便捷字段，部分中转也会带
    if isinstance(ot, str) and ot.strip():
        return ot
    text = ""
    out = data.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue  # 跳过 reasoning 等其它条目
            content = item.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") in ("output_text", "text"):
                        text += blk.get("text", "")
            elif isinstance(content, str):
                text += content
    if text.strip():
        return text
    # 个别中转把 responses 也包成 chat 形状，兜底再试一次
    return _extract_chat_text(data, _strict=False)


def _extract_chat_text(data: dict, *, _strict: bool = True) -> str:
    """解析 OpenAI Chat Completions 返回的文本。_strict=False 时失败返回空串而非抛错。"""
    choices = data.get("choices") or []
    if not choices:
        if not _strict:
            return ""
        # 兜底：也许其实是 responses 形状
        t = _extract_responses_text(data)
        if t:
            return t
        raise ChartError("上游无 choices 返回")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):  # 某些实现返回分段
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


async def _post_json(url: str, body: dict, headers: dict, *, label: str) -> dict:
    """统一的带重试 POST，返回解析后的 JSON；失败抛 ChartError。"""
    last_err = "未知错误"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"连接失败: {e}"
                if attempt < _HTTP_RETRIES:
                    continue
                raise ChartError(last_err)
            if resp.status_code in (502, 503, 504):
                last_err = f"{label} {resp.status_code}: {resp.text[:300]}"
                if attempt < _HTTP_RETRIES:
                    continue
                raise ChartError(last_err)
            if resp.status_code >= 400:
                log.error("chart %s %s: %s", label, resp.status_code, resp.text[:800])
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                except Exception:
                    msg = resp.text[:300]
                raise ChartError(f"{label} {resp.status_code}: {msg}")
            try:
                return resp.json()
            except Exception:
                raise ChartError("上游返回非 JSON")
    raise ChartError(last_err)


async def _call_llm(messages: list[dict]) -> str:
    """调一次公式/图表模型，返回文本。按 api_style 选择 Chat Completions 或 Responses API。"""
    cfg = await db.get_provider("chart")
    if not cfg["base"] or not cfg["key"]:
        raise ChartError("公式/图表模型未配置（管理面板 → AI 提供商 → 公式图表）")
    headers = {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}
    style = _resolve_style(cfg)

    if style == "responses":
        # OpenAI Responses API：system 用 instructions，对话用 input；输出在 output[].content[].text
        url = f"{cfg['base']}/responses"
        body = {
            "model": cfg["model"],
            "instructions": _SYSTEM_PROMPT,
            "input": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_output_tokens": 8192,  # 推理模型会先耗 reasoning token，给足避免截断
        }
        data = await _post_json(url, body, headers, label="responses")
        content = _extract_responses_text(data)
    else:
        url = f"{cfg['base']}/chat/completions"
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            "max_tokens": 4096,
        }
        data = await _post_json(url, body, headers, label="chat")
        content = _extract_chat_text(data)

    if not content:
        log.error("chart 上游返回不含文本 style=%s data=%s", style, str(data)[:500])
        raise ChartError("上游返回不含文本内容")
    return content


async def generate_chart(prompt: str, *, max_repairs: int = 1) -> tuple[bytes, dict]:
    """让模型产出 HTML 正文片段，套模板成自包含 HTML。返回 (html_bytes, meta)。

    不再服务器端执行/渲染——HTML 由浏览器端渲染（公式/流程图）。模型没给出有效
    HTML 时最多再要一次。全部失败抛 ChartError。
    """
    messages: list[dict] = [{"role": "user", "content": f"需求：{prompt.strip()}"}]
    last_err = "未知错误"
    for attempt in range(max_repairs + 1):
        text = await _call_llm(messages)
        try:
            frag = extract_html(text)
        except ChartError as e:
            last_err = str(e)
            log.warning("chart 未取得 HTML attempt=%d: %s", attempt + 1, last_err)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": _repair_prompt()})
            continue
        html = wrap_html(frag)
        log.info("chart 生成成功 attempt=%d html_bytes=%d", attempt + 1, len(html))
        return html.encode("utf-8"), {"format": "html"}
    raise ChartError(f"多次尝试仍未获得 HTML：{last_err}")
