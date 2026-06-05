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
- 流程图 / 时序图 / 状态图 / 类图 / 甘特图 / ER 图等**关系类图**：用 Mermaid，写成
  <pre class="mermaid"> ... </pre>，例如
  <pre class="mermaid">graph TD; A[开始] --> B{判断}; B -->|是| C[执行]; B -->|否| D[结束];</pre>
- 文字排版：用语义化标签 h1/h2/h3/p/ul/ol/table/figure/figcaption/blockquote/code 等。
- 用一个 <h1> 作为整篇标题；需要时给表格加 <caption>。可配合小标题、要点列表、数据表把内容讲清楚。

# 选型建议
- 有数值 / 趋势 / 占比 / 分布 → ECharts（最好看）。关系 / 流程 / 步骤 → Mermaid。公式 → MathJax。

# 硬性约定
- 只输出 <body> 内部的 HTML 片段；**不要**写 <html>/<head>/<body>/<script>/<style>/<link>。
  CSS、ECharts、MathJax、Mermaid 都由外层模板统一提供，你只负责内容与结构。
- ECharts 的 option 必须是能被 JSON.parse 的纯 JSON（不能写 JS 表达式 / 函数 / 变量）。
- 不要引用任何外部图片/字体/JS；不要写 onclick 等事件属性。
- 正文用中文；公式、变量、代码保持原样。

# 输出
直接输出 HTML 片段本身（可放在 ```html 代码块里），不要任何额外解释。
"""


def _repair_prompt() -> str:
    return ('请只输出 <body> 内的 HTML 正文片段：数据图表用 '
            '<div class="echarts" data-h="360">{合法的 ECharts option JSON}</div>，'
            '公式用 MathJax，流程图用 <pre class="mermaid">。'
            '不要 <html>/<head>/<script>/<style>，不要解释。')


def extract_html(text: str) -> str:
    """从模型回复里取出 HTML 正文片段，并剔除模型违规塞进来的 script/style/外链。"""
    if not text:
        raise ChartError("返回内容为空")
    m = re.search(r"```(?:html?)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    frag = (m.group(1) if m else text).strip()
    bm = re.search(r"<body[^>]*>(.*)</body>", frag, flags=re.DOTALL | re.IGNORECASE)
    if bm:
        frag = bm.group(1).strip()
    frag = re.sub(r"</?(?:html|head|body)[^>]*>", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"<script\b.*?</script>", "", frag, flags=re.DOTALL | re.IGNORECASE)
    frag = re.sub(r"<style\b.*?</style>", "", frag, flags=re.DOTALL | re.IGNORECASE)
    frag = re.sub(r"<link\b[^>]*>", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", frag, flags=re.IGNORECASE)  # 去事件属性
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
  window.__renderReady=(async function(){
    try{_initMetrics();}catch(e){}
    await _waitFor(function(){return window.echarts;},9000);
    try{_initEcharts();}catch(e){}
    await _waitFor(function(){return window.mermaid;},9000);
    try{mermaid.initialize({startOnLoad:false,theme:'neutral',securityLevel:'strict',htmlLabels:false,flowchart:{htmlLabels:false}});await mermaid.run();}catch(e){}
    await _waitFor(function(){return window.MathJax&&MathJax.startup&&MathJax.startup.promise;},9000);
    try{await MathJax.startup.promise;}catch(e){}
    try{if(window.MathJax&&MathJax.typesetPromise)await MathJax.typesetPromise();}catch(e){}
  })();
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
