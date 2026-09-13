// Service Worker：扩展的唯一"大脑"。
//
// 分工：
//   background.js（这里）= 与本程序的本机 WebSocket 通道 + 状态机 + 指令执行
//   offscreen.js        = 取标签页音频、回放给用户、出 PCM（MV3 里只有它能用 Web Audio）
//   popup.js            = 用户界面（选标签页、开始/停止、看状态）
//
// 为什么 WS 放在 SW 而不是 offscreen：
//   本程序要能"随时"知道扩展在不在、当前在送哪个标签页。SW 是常驻组件，
//   而 offscreen 文档只在采集期间存在。放 SW 里，连接的生命周期就是扩展的生命周期。
//
// 音频路径（每个箭头都是内存里的一次拷贝/转移，20ms 一块，约 3.8KB）：
//   标签页 → tabCapture 流 → offscreen（同时接回扬声器）→ AudioWorklet → SW → ws → 本程序

const DEFAULT_PORT = 38991;
const PATH = "/lst/tab";
const PROTOCOL = 1;
const FRAMES_PER_CHUNK = 960; // 20ms @48k，立体声 float32 = 7680 字节/帧
const OUT_CHANNELS = 2;
const STOP_WITHOUT_APP_MS = 8000; // 与本程序断线后，等这么久还没回来就停止采集

let ws = null;
let connecting = false;
let retryMs = 1000;
let retryTimer = null;
let offscreenPort = null;
let offscreenWaiter = null;
let capturing = null; // {tabId, title, url, rate, channels}
let lastError = "";
let lastErrorHint = "";
let stopTimer = null;
let booted = false;
let pendingCapture = false;
let pendingTabId = null;
/** 本程序请求过采集、但还没被用户授权（浏览器按**标签页**授予 kTabCaptureForTab）。
 *  这个标记的用途：等用户在某个标签页上点扩展图标时，弹出窗一打开就自动开工，
 *  用户不必再点第二个按钮。 */

function log(...args) {
  console.log("[lst-sw]", ...args);
}

// --------------------------------------------------------------------------- //
// 设置与常量
// --------------------------------------------------------------------------- //
async function getSettings() {
  return await chrome.storage.local.get({ port: DEFAULT_PORT, token: "" });
}

/** 测试/自检用的开发配置：扩展目录里放一个 dev.json 才会生效，正常安装没有这个文件。 */
async function getDevConfig() {
  try {
    const r = await fetch(chrome.runtime.getURL("dev.json"));
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab || null;
}

// --------------------------------------------------------------------------- //
// WebSocket
// --------------------------------------------------------------------------- //
function wsReady() {
  return ws && ws.readyState === WebSocket.OPEN;
}

function send(obj) {
  if (!wsReady()) return false;
  try {
    ws.send(JSON.stringify(obj));
    return true;
  } catch (e) {
    log("发送失败", e);
    return false;
  }
}

async function connect(reason) {
  // 三道闸门缺一不可：
  //   1) connecting：connect() 是异步的（要先读设置），不加锁就会被并发调用创建出多条连接；
  //   2) wsOpen/wsConnecting：已有连接就不要再来一条。
  // 曾经因为漏了第 1 条 + onclose 里无条件置空 ws，导致"新连接顶掉旧连接 →
  // 旧连接的 onclose 又触发重连 → 再顶掉"，每秒互相踢一次，永远握不上手。
  if (connecting) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  connecting = true;
  let sock = null;
  try {
    const { port } = await getSettings();
    log(`开始连接 ws://127.0.0.1:${port}${PATH}（${reason || ""}）`);
    sock = new WebSocket(`ws://127.0.0.1:${port}${PATH}`);
  } catch (e) {
    log("创建连接失败", e);
    connecting = false;
    scheduleRetry();
    return;
  }
  connecting = false;
  ws = sock;

  sock.onopen = async () => {
    retryMs = 1000;
    log("已连接", reason || "");
    await reportCommands();
    await sendHello();
    announce();
  };
  sock.onmessage = (ev) => {
    let msg = null;
    try {
      msg = JSON.parse(ev.data);
    } catch (e) {
      return;
    }
    handleCommand(msg);
  };
  sock.onerror = () => {};
  sock.onclose = () => {
    // 只在"当前连接就是这条"时才清空——否则会把已经接班的新连接一起清掉
    if (ws === sock) {
      ws = null;
      announce();
      scheduleRetry();
    }
    // 与本程序失联：不是立刻停采集（用户可能只是重启了字幕程序），
    // 但也不能无限期占着标签页音频不放
    if (capturing && !stopTimer) {
      stopTimer = setTimeout(async () => {
        stopTimer = null;
        if (!wsReady() && capturing) {
          log("与本程序失联过久，停止采集");
          await stopCapture("app-gone");
        }
      }, STOP_WITHOUT_APP_MS);
    }
  };
}

function scheduleRetry() {
  if (retryTimer) return;
  const delay = retryMs;
  retryMs = Math.min(retryMs * 2, 30000);
  retryTimer = setTimeout(() => {
    retryTimer = null;
    connect();
  }, delay);
}

async function sendHello() {
  const { token } = await getSettings();  let tab = { id: 0, title: "", url: "" };
  if (capturing) {
    tab = { id: capturing.tabId, title: capturing.title || "", url: capturing.url || "" };
  } else {
    const at = await activeTab().catch(() => null);
    if (at) tab = { id: at.id || 0, title: at.title || "", url: at.url || "" };
  }
  send({
    type: "hello",
    protocol: PROTOCOL,
    role: "control",
    token,
    format: { rate: 48000, channels: OUT_CHANNELS, dtype: "float32" },
    tab,
    browser: { name: browserName(), version: navigator.userAgent },
    capturing: !!capturing,
  });
}

function browserName() {
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return "Edge";
  if (/Chrome\//.test(ua)) return "Chrome";
  return "Chromium 浏览器";
}

function error(code, message, hint) {
  lastError = message;
  lastErrorHint = hint || "";
  log("错误", code, message);
  send({ type: "error", code, message, hint: hint || "" });
  announce();
}

// --------------------------------------------------------------------------- //
// 来自本程序的指令
// --------------------------------------------------------------------------- //
async function handleCommand(msg) {
  const kind = msg.type;
  if (kind === "capture") {
    await startCapture(msg.tabId || null);
  } else if (kind === "stop") {
    await stopCapture("app-request");
  } else if (kind === "state") {
    await sendHello();
    sendStatus();
  }
}

// --------------------------------------------------------------------------- //
// offscreen 文档
// --------------------------------------------------------------------------- //
async function ensureOffscreen() {
  if (!(await chrome.offscreen.hasDocument())) {
    try {
      await chrome.offscreen.createDocument({
        url: "offscreen.html",
        reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
        justification: "读取标签页音频并回放，供本机字幕程序使用",
      });
    } catch (e) {
      error("offscreen-create", `创建 offscreen 文档失败：${(e && e.message) || e}`);
      return null;
    }
  }
  const port = await waitForOffscreenPort(6000);
  if (!port) {
    error("offscreen-port", "offscreen 文档已创建，但它没有连上后台（可能被 CSP 拦住了）");
  }
  return port;
}

function waitForOffscreenPort(timeoutMs) {
  if (offscreenPort) return offscreenPort;
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      offscreenWaiter = null;
      resolve(offscreenPort);
    }, timeoutMs);
    offscreenWaiter = (port) => {
      clearTimeout(timer);
      offscreenWaiter = null;
      resolve(port);
    };
  });
}

function ask(port, msg) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("等待 offscreen 响应超时")), 8000);
    const listener = (reply) => {
      clearTimeout(timer);
      port.onMessage.removeListener(listener);
      resolve(reply);
    };
    port.onMessage.addListener(listener);
    port.postMessage(msg);
  });
}

// --------------------------------------------------------------------------- //
// 采集
// --------------------------------------------------------------------------- //
async function startCapture(tabId) {
  lastError = "";
  lastErrorHint = "";
  pendingCapture = true;
  const targetId = tabId || (await activeTab())?.id;
  pendingTabId = targetId || null;
  if (!targetId) {
    error("no-tab", "找不到要采集的标签页", "请先在浏览器里打开要字幕的那个标签页");
    return;
  }
  const tab = await chrome.tabs.get(targetId).catch(() => null);

  let streamId;
  try {
    streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: targetId });
  } catch (e) {
    // 浏览器按**标签页**授予 kTabCaptureForTab：必须由用户在该标签页上
    // 点扩展图标（或按 action 快捷键）触发过，才允许取它的音频。
    // 这不是 bug，是安全策略——把话说清楚，并给出唯一可操作的那一步。
    error(
      "gesture",
      `浏览器要求先由你触发一次（${(e && e.message) || e}）`,
      "切到你要字幕的那个标签页，点一下工具栏里的扩展图标（或按 Ctrl+Shift+U）。" +
        "扩展会自动开始；以后换标签页需要再点一次。"
    );
    return;
  }

  const port = await ensureOffscreen();
  if (!port) {
    return; // ensureOffscreen 已经报过具体原因
  }

  let reply;
  try {
    const { token, port: wsPort } = await getSettings();
    reply = await ask(port, {
      type: "start",
      streamId,
      tabId: targetId,
      title: tab ? tab.title : "",
      url: tab ? tab.url : "",
      framesPerChunk: FRAMES_PER_CHUNK,
      // offscreen 会拿这两个值自己连本程序送音频（不再经 SW 转发）
      wsPort,
      token,
    });
  } catch (e) {
    error("offscreen-failed", String((e && e.message) || e));
    return;
  }
  if (!reply || reply.kind !== "started") {
    error("offscreen-failed", (reply && reply.message) || "offscreen 未启动成功");
    return;
  }

  capturing = {
    tabId: targetId,
    title: reply.title || (tab ? tab.title : ""),
    url: reply.url || (tab ? tab.url : ""),
    rate: reply.rate,
    channels: reply.channels,
  };
  pendingCapture = false; // 已经开工，不用再等用户那一下了
  pendingTabId = null;
  await chrome.storage.local.set({ lastTabId: targetId });
  await chrome.storage.session.set({ capturing });

  // 实际格式由**音频连接**的 hello 上报（它才是送 PCM 的那条），
  // 这里只更新控制连接上的状态。
  sendStatus();
  setBadge("●");
  announce();
  log(`开始采集 tab=${targetId} ${capturing.rate}Hz/${capturing.channels}ch`);
}

/**
 * 音轨结束后的自动恢复。
 *
 * 为什么需要：``chrome.tabCapture`` 的音频音轨会在目标标签页"不再出声"时结束
 * （视频播完 / 暂停 / 播放器换曲），以前我们收到 ended 就直接停止采集，
 * 用户必须再点一次扩展图标才能继续——实测反馈就是"发送会暂停，还得手动再点一次"。
 *
 * 自动重试不需要用户再授权（那次授权还在这个标签页上），所以绝大多数情况能自愈；
 * 只有标签页被关掉、或浏览器要求重新触发时，才去求人（并把原因讲清楚）。
 */
async function resumeAfterTrackEnded(tabId) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    await new Promise((r) => setTimeout(r, 1200));
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    if (!tab) {
      send({ type: "log", message: `标签页 ${tabId} 已关闭，停止采集` });
      await stopCapture("tab-closed");
      return;
    }
    send({ type: "log", message: `音轨结束，自动尝试恢复采集（第 ${attempt}/3 次）` });
    await startCapture(tabId);
    if (capturing) {
      send({ type: "log", message: "已自动恢复采集，无需手动再点" });
      return;
    }
  }
  error(
    "resume-failed",
    "音轨结束后自动恢复失败",
    "请在浏览器里再点一次扩展图标（或按 Ctrl+Shift+U）"
  );
}

async function stopCapture(reason) {  if (stopTimer) {
    clearTimeout(stopTimer);
    stopTimer = null;
  }
  const port = offscreenPort;
  if (port) {
    try {
      await ask(port, { type: "stop" });
    } catch (e) {
      log("停止时未收到 offscreen 确认", e);
    }
  }
  try {
    if (await chrome.offscreen.hasDocument()) await chrome.offscreen.closeDocument();
  } catch (e) {
    log("关闭 offscreen 失败", e);
  }
  offscreenPort = null;
  capturing = null;
  pendingCapture = false;
  pendingTabId = null;
  await chrome.storage.session.remove("capturing");
  setBadge("");
  send({ type: "status", capturing: false, tab: { id: 0, title: "", url: "" } });
  announce();
  log("已停止采集", reason);
}

function sendStatus() {
  if (capturing) {
    send({
      type: "status",
      capturing: true,
      tab: { id: capturing.tabId, title: capturing.title, url: capturing.url },
    });
  } else {
    send({ type: "status", capturing: false, tab: { id: 0, title: "", url: "" } });
  }
}

function setBadge(text) {
  try {
    chrome.action.setBadgeText({ text });
    chrome.action.setBadgeBackgroundColor({ color: "#2e7d32" });
  } catch (e) {
    /* 某些环境不支持徽标，忽略 */
  }
}

// offscreen → SW：只有控制/状态（音频已改由 offscreen 直连本程序）
let msgsSeen = 0;
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "lst-audio") return;
  offscreenPort = port;
  log(`offscreen 已连上（端口 ${port.name}）`);
  if (offscreenWaiter) offscreenWaiter(port);
  port.onMessage.addListener((msg) => {
    if (msgsSeen < 4) {
      msgsSeen += 1;
      log(`SW 收到端口消息 #${msgsSeen}：type=${msg && msg.type}`);
    }
    if (msg.type === "ended") {
      log("目标标签页的音频结束了");
      // 音轨结束 ≠ 用户想停：最常见的场景是"视频播完了 / 暂停了一下"，
      // 以前这里直接 stopCapture，用户就得再点一次扩展图标。
      // 现在自动重试几次（授权仍在，不用用户再点），真的不行才去求人。
      if (capturing && capturing.tabId) {
        resumeAfterTrackEnded(capturing.tabId);
      } else {
        stopCapture("track-ended");
      }
    } else if (msg.type === "failed") {
      error("offscreen-failed", msg.message || "offscreen 报错");
    }
  });
  port.onDisconnect.addListener(() => {
    if (offscreenPort === port) offscreenPort = null;
  });
});

// popup → SW
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    if (msg.type === "get-state") {
      sendResponse(await stateSnapshot());
    } else if (msg.type === "start-capture") {
      await startCapture(msg.tabId || null);
      sendResponse(await stateSnapshot());
    } else if (msg.type === "stop-capture") {
      await stopCapture("popup");
      sendResponse(await stateSnapshot());
    } else if (msg.type === "save-settings") {
      const patch = {};
      if (msg.port) patch.port = Number(msg.port);
      if (typeof msg.token === "string") patch.token = msg.token;
      await chrome.storage.local.set(patch);
      // 关掉现有连接让 onclose 去重连（立刻重试一次，别让用户等退避）
      retryMs = 300;
      if (stopTimer) {
        clearTimeout(stopTimer);
        stopTimer = null;
      }
      if (ws) {
        try {
          ws.close();
        } catch (e) {}
      }
      sendResponse(await stateSnapshot());
    } else {
      sendResponse({ ok: false, message: "未知消息" });
    }
  })();
  return true; // 异步 sendResponse
});

async function stateSnapshot() {
  const { port, token } = await getSettings();
  return {
    ok: true,
    connected: wsReady(),
    port,
    hasToken: !!token,
    capturing,
    pending: pendingCapture,
    pendingTabId,
    lastError,
    lastErrorHint,
    browser: browserName(),
    version: chrome.runtime.getManifest().version,
  };
}

function announce() {
  stateSnapshot().then((state) => {
    chrome.runtime.sendMessage({ type: "state", state }).catch(() => {});
  });
}

// --------------------------------------------------------------------------- //
// 启动
// --------------------------------------------------------------------------- //
async function restoreSession() {
  const saved = await chrome.storage.session.get({ capturing: null });
  if (saved.capturing) {
    // SW 被回收过：offscreen 文档可能还在，但端口丢了；保守起见停掉重来
    log("发现残留的采集状态，清理", saved.capturing.tabId);
    await stopCapture("stale");
  }
}

async function devAutoStart() {
  const dev = await getDevConfig();
  if (!dev || !dev.autoStart) return;
  log("dev.json 触发了自动采集", JSON.stringify(dev));
  for (let i = 0; i < 25; i++) {
    const tabs = await chrome.tabs.query({});
    const want = dev.autoStartTabTitle || "";
    const hit = tabs.find((t) => {
      const title = t.title || "";
      if (want) return title.startsWith(want);
      return !!t.audible;
    });
    if (hit) {
      await startCapture(hit.id);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  log("dev.json 自动采集超时：没找到目标标签页");
}

// --------------------------------------------------------------------------- //
// 用户亲手触发（快捷键）
// --------------------------------------------------------------------------- //
// 浏览器规定：取标签页音频前，扩展必须被用户"触发"过（activeTab 授权）。
// 所以这个快捷键不是锦上添花，而是**保证能开工**的那条路——
// 命令行/程序都无法代替用户按下它。按下后授权只给"当前活动标签页"，
// 因此要字幕哪个标签页，就得先切到那个标签页再按。
chrome.commands.onCommand.addListener(async (command) => {
  send({ type: "log", message: `收到快捷键触发：${command}` });
  if (command !== "lst-toggle") return;
  log("收到快捷键触发");
  if (capturing) {
    await stopCapture("command-toggle");
    return;
  }
  const tab = await activeTab();
  await startCapture(tab ? tab.id : null);
});

chrome.alarms.create("lst-keepalive", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "lst-keepalive") connect("alarm");
});

async function bootstrap(reason) {
  if (booted) return;
  booted = true;
  await restoreSession();
  await connect(reason);
  devAutoStart();
  announce();
}

/** 把"快捷键到底注册成了什么"报上去：suggested_key 可能被浏览器占用/拒绝，
 *  那时 onCommand 永远不会触发，界面上却看不出原因。必须在连接**建立之后**发，
 *  否则消息会被丢掉（connect() 不等 onopen）。 */
async function reportCommands() {
  try {
    const cmds = await chrome.commands.getAll();
    send({ type: "log", message: `快捷键注册情况：${JSON.stringify(cmds)}` });
  } catch (e) {
    send({ type: "log", message: `读取快捷键失败：${e}` });
  }
}

chrome.runtime.onStartup.addListener(() => bootstrap("startup"));
chrome.runtime.onInstalled.addListener(() => bootstrap("installed"));

// SW 每次被唤醒都要做的事（模块顶层只会在 SW 启动时跑一次）
bootstrap("sw-start");
