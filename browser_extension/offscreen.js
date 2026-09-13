// offscreen 文档：唯一有 DOM / Web Audio 的地方。
//
// 职责：
//   1. 用 chrome.tabCapture 给过来的 streamId 打开标签页音频流；
//   2. **把音频接回扬声器**——tabCapture 会把标签页音频从默认输出"摘走"，
//      不接回去用户就听不见了（这是最容易忘、后果最严重的一步）；
//   3. 另分一路给 AudioWorklet，攒成 20ms 的 float32 交错 PCM；
//   4. **自己开一条 WebSocket 把 PCM 直接送给本程序**。
//
// 为什么音频不经过 service worker 转发：
//   实测（2026-09）PCM 走 ``chrome.runtime`` 端口的 ArrayBuffer 那一跳会变成 ``null``，
//   于是 ``ws.send(null)`` 发出去的是字符串 "[object Object]"，
//   程序侧只看到"无法解析的控制消息"、一个字节的音频都收不到。
//   直连还顺带省掉一次拷贝、少一层延迟。
//   SW 那条连接只负责控制与状态（角色 role=control），音频这条是 role=audio。

let port = null;        // 与 service worker 的长连接（控制）
let sock = null;        // 直连本程序的 WebSocket（音频）
let ctx = null;         // AudioContext
let stream = null;      // 标签页音频流
let source = null;      // MediaStreamAudioSourceNode
let worklet = null;     // AudioWorkletNode
let silentGain = null;  // 让 worklet 挂进渲染图（否则可能不被驱动）
let current = null;     // { tabId, title, url, rate, channels }
let sent = 0;
let sendTimer = null;   // "socket 没开时先攒着"的定时器（简单起见：直接丢，见下）
let reconnect = null;

function log(...args) {
  console.log("[lst-offscreen]", ...args);
}

// --------------------------------------------------------------------------- //
// 与 service worker 的控制通道
// --------------------------------------------------------------------------- //
function sw() {
  if (!port) {
    port = chrome.runtime.connect({ name: "lst-audio" });
    // ⚠️ 监听必须挂在**自己 connect 出来的这个 port** 上：
    // chrome.runtime.onConnect 只在"有别人连进来"时触发，而 offscreen 文档
    // 永远只是主动连接的一方——挂在 onConnect 上等于没人听。
    port.onMessage.addListener((msg) => {
      handleMessage(msg, port);
    });
    port.onDisconnect.addListener(() => {
      port = null;
    });
  }
  return port;
}

async function handleMessage(msg, replyPort) {
  try {
    if (msg.type === "start") {
      const info = await startCapture(msg);
      replyPort.postMessage({ type: "started", ...info });
    } else if (msg.type === "stop") {
      const was = await stopCapture("requested");
      replyPort.postMessage({ type: "stopped", tabId: was ? was.tabId : null });
    } else if (msg.type === "stats") {
      replyPort.postMessage({ type: "stats", sent, tabId: current ? current.tabId : null });
    }
  } catch (err) {
    log("处理失败", err);
    replyPort.postMessage({ type: "failed", message: String((err && err.message) || err) });
  }
}

// 载入即连：service worker 创建完 offscreen 文档后会等这条连接来派发任务
sw();

// --------------------------------------------------------------------------- //
// 音频 WebSocket（直连本程序）
// --------------------------------------------------------------------------- //
function openAudioSocket(wsPort, token, fmt, tab) {
  return new Promise((resolve, reject) => {
    const url = `ws://127.0.0.1:${wsPort}/lst/tab`;
    let s;
    try {
      s = new WebSocket(url);
    } catch (e) {
      reject(e);
      return;
    }
    const timer = setTimeout(() => {
      reject(new Error(`连接 ${url} 超时`));
    }, 4000);

    s.onopen = () => {
      clearTimeout(timer);
      s.send(
        JSON.stringify({
          type: "hello",
          protocol: 1,
          role: "audio",
          token: token || "",
          format: fmt,
          tab,
          browser: { name: "Edge/Chrome" },
          capturing: true,
        })
      );
      log(`音频通道已连上 ${url}`);
      resolve(s);
    };
    s.onerror = () => {
      clearTimeout(timer);
      reject(new Error(`连不上 ${url}（本程序里的「浏览器标签页」模式没开？）`));
    };
    s.onclose = () => {
      if (sock === s) {
        sock = null;
        log("音频通道断开");
        scheduleReconnect();
      }
    };
  });
}

/** 与本程序失联时（例如字幕程序重启）继续尝试，采集中断不掉。 */
function scheduleReconnect() {
  if (reconnect || !current) return;
  reconnect = setTimeout(async () => {
    reconnect = null;
    if (!current || sock) return;
    try {
      sock = await openAudioSocket(current.wsPort, current.token, fmtOf(current), tabOf(current));
      log("音频通道已重连");
    } catch (e) {
      log("音频通道重连失败", String(e && e.message));
      scheduleReconnect();
    }
  }, 2000);
}

function fmtOf(c) {
  return { rate: c.rate, channels: c.channels, dtype: "float32" };
}

function tabOf(c) {
  return { id: c.tabId, title: c.title || "", url: c.url || "" };
}

// --------------------------------------------------------------------------- //
// 采集
// --------------------------------------------------------------------------- //
async function stopCapture(reason) {
  try {
    if (worklet) {
      worklet.port.postMessage({ type: "flush" });
      worklet.disconnect();
    }
  } catch (e) {}
  try { if (source) source.disconnect(); } catch (e) {}
  try { if (silentGain) silentGain.disconnect(); } catch (e) {}
  try { if (stream) stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
  try { if (ctx) await ctx.close(); } catch (e) {}
  try { if (sock) sock.close(); } catch (e) {}
  if (reconnect) {
    clearTimeout(reconnect);
    reconnect = null;
  }
  const was = current;
  worklet = source = silentGain = stream = ctx = sock = null;
  current = null;
  if (was) log("已停止采集", reason || "");
  return was;
}

async function startCapture(msg) {
  await stopCapture("restart");
  const { streamId, tabId, title, url, framesPerChunk, wsPort, token } = msg;
  const channels = 2;

  if (!wsPort) throw new Error("没有拿到本程序的端口（请先在程序里选「浏览器标签页」并开始）");

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  ctx = new AudioContext({ sampleRate: 48000 });
  await ctx.resume().catch(() => {});
  const rate = Math.round(ctx.sampleRate) || 48000;
  log(`AudioContext state=${ctx.state} rate=${rate}`);
  if (ctx.state !== "running") {
    // 自动播放策略可能把上下文挂在 suspended：那时 worklet 一次都不会被驱动，
    // 表现为"扩展说开始采集了，程序一个字节都收不到"。
    throw new Error(`音频上下文被浏览器挂起（state=${ctx.state}）`);
  }

  current = { tabId, title: title || "", url: url || "", rate, channels, wsPort, token };
  sock = await openAudioSocket(wsPort, token, fmtOf(current), tabOf(current));

  source = ctx.createMediaStreamSource(stream);

  // ① 原样回放给用户听
  source.connect(ctx.destination);

  // ② 另分一路做分析（worklet 输出接 0 增益，保证它一定被音频线程驱动）
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  worklet = new AudioWorkletNode(ctx, "lst-pcm", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { channels, framesPerChunk: framesPerChunk || 960 },
  });

  let firstPcm = false;
  worklet.port.onmessage = (e) => {
    const data = e.data;
    if (data && data.type === "alive") {
      if (!firstPcm) log(`worklet 已被音频线程驱动：calls=${data.calls} 输入声道=${data.channels}`);
      return;
    }
    const buf = data;
    if (!buf || typeof buf.byteLength !== "number") return;
    if (!firstPcm) {
      firstPcm = true;
      log(`收到第一块 PCM：${buf.byteLength} 字节`);
    }
    sent += buf.byteLength;
    if (sock && sock.readyState === WebSocket.OPEN) {
      sock.send(buf); // ArrayBuffer → 二进制帧，直连
    }
    // socket 没开就丢掉这块（字幕是实时消费品，攒着只会越积越旧）
  };
  source.connect(worklet);
  silentGain = ctx.createGain();
  silentGain.gain.value = 0;
  worklet.connect(silentGain).connect(ctx.destination);

  // 目标标签页被关掉 / 声音结束
  stream.getAudioTracks().forEach((track) => {
    track.addEventListener("ended", () => {
      log("音轨结束");
      sw().postMessage({ type: "ended" });
    });
  });

  log(`开始采集 tab=${tabId} rate=${rate} channels=${channels}`);
  return { kind: "started", rate, channels, tabId, title: current.title, url: current.url };
}
