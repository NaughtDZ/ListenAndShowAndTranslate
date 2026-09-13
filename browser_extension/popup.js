// popup：选标签页、开始/停止、看状态。
//
// 这里是**必须由用户亲手点**的地方：浏览器规定"取标签页音频"要由用户触发过扩展
// 才允许，所以「开始字幕」这个按钮不是可选项，而是安全策略要求的那一下。

const $ = (id) => document.getElementById(id);

let state = null;
let tabs = [];
let selectedId = null;
let timer = null;
let autoTried = false;

async function call(msg) {
  try {
    return await chrome.runtime.sendMessage(msg);
  } catch (e) {
    return { ok: false, message: String((e && e.message) || e) };
  }
}

function shortUrl(url) {
  try {
    const u = new URL(url);
    return u.hostname.replace(/^www\./, "");
  } catch (e) {
    return "";
  }
}

async function loadTabs() {
  const audible = await chrome.tabs.query({ audible: true });
  const [active] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  const saved = (await chrome.storage.local.get({ lastTabId: 0 })).lastTabId;

  const seen = new Set();
  tabs = [];
  const push = (t, tag) => {
    if (!t || seen.has(t.id)) return;
    seen.add(t.id);
    tabs.push({
      id: t.id,
      title: t.title || "(无标题)",
      url: t.url || "",
      favIconUrl: t.favIconUrl || "",
      tag,
    });
  };
  audible.forEach((t) => push(t, "正在发声"));
  push(active, "当前");
  // 目标标签页平时可能是静音的（还没开始播），所以也给几个静音标签页兜底
  const others = await chrome.tabs.query({ currentWindow: true });
  others.forEach((t) => push(t, ""));

  if (!tabs.some((t) => t.id === selectedId)) {
    const prefer = tabs.find((t) => t.id === saved) || tabs.find((t, i) => i === 0);
    selectedId = prefer ? prefer.id : null;
  }
  // 正在采集的那个优先选中，避免用户误以为没在送
  if (state && state.capturing && state.capturing.tabId) selectedId = state.capturing.tabId;
}

function renderTabs() {
  const box = $("list");
  box.innerHTML = "";
  if (!tabs.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "这个窗口里没有可选的标签页。";
    box.appendChild(empty);
    return;
  }
  for (const t of tabs) {
    const row = document.createElement("div");
    row.className = "item" + (t.id === selectedId ? " selected" : "");

    const img = document.createElement("img");
    if (t.favIconUrl) img.src = t.favIconUrl;
    row.appendChild(img);

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = t.title;
    name.title = t.title;
    row.appendChild(name);

    const tag = document.createElement("div");
    tag.className = "tag";
    tag.textContent = t.tag || shortUrl(t.url);
    row.appendChild(tag);

    row.addEventListener("click", () => {
      selectedId = t.id;
      chrome.storage.local.set({ lastTabId: t.id });
      renderTabs();
      renderButtons();
    });
    box.appendChild(row);
  }
}

function renderStatus() {
  const dot = $("dot");
  const status = $("status");
  if (!state) {
    dot.className = "dot";
    status.textContent = "正在检查…";
    return;
  }
  if (state.capturing) {
    dot.className = "dot on";
    const title = state.capturing.title || `标签页 ${state.capturing.tabId}`;
    status.textContent = `正在发送：${title}${
      state.capturing.rate ? `（${state.capturing.rate}Hz）` : ""
    }`;
  } else if (state.connected) {
    dot.className = "dot on";
    status.textContent = `已连上「听·显·译」（端口 ${state.port}）。选一个标签页点开始。`;
  } else {
    dot.className = "dot off";
    status.textContent = `没连上「听·显·译」（试的是 127.0.0.1:${state.port}）。请先在本程序里选「浏览器标签页」并点开始。`;
  }

  const hint = $("hint");
  if (state.lastError) {
    hint.hidden = false;
    hint.textContent = state.lastErrorHint
      ? `${state.lastErrorHint}\n\n（技术细节：${state.lastError}）`
      : state.lastError;
  } else {
    hint.hidden = true;
    hint.textContent = "";
  }
}

function renderButtons() {
  const busy = !!(state && state.capturing);
  $("stop").disabled = !busy;
  $("start").disabled = !selectedId;
  $("start").textContent = busy ? "换成这个标签页" : "开始字幕";
  $("port").value = state ? state.port : "";
  $("token").placeholder = state && state.hasToken ? "已设置（留空不改）" : "留空表示不校验";
}

async function refresh() {
  state = await call({ type: "get-state" });
  await loadTabs();
  renderTabs();
  renderStatus();
  renderButtons();
  await maybeAutoStart();
}

/**
 * 本程序已经在等音频了，而且用户刚刚点开这个弹窗（= 他刚刚触发了扩展，
 * 浏览器这才把当前标签页的采集权限放出来）——那就别再让用户点第二次按钮。
 *
 * 注意：授权是**按标签页**给的，只给弹窗打开时的活动标签页，
 * 所以这里一律用活动标签页，而不是他上次选过的那个。
 */
async function maybeAutoStart() {
  if (autoTried || !state || !state.pending || state.capturing) return;
  autoTried = true;
  const [active] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!active) return;
  const changed = state.pendingTabId && state.pendingTabId !== active.id;
  state = await call({ type: "start-capture", tabId: active.id });
  if (state && state.capturing) {
    const hint = $("hint");
    if (changed) {
      hint.hidden = false;
      hint.textContent =
        "授权只对「点开扩展时所在的那个标签页」有效，所以这次开始的是当前标签页。" +
        "要换成别的标签页：先切过去，再点一次扩展图标。";
    }
  }
  renderStatus();
  renderButtons();
}

$("start").addEventListener("click", async () => {
  const btn = $("start");
  btn.disabled = true;
  btn.textContent = "正在启动…";
  state = await call({ type: "start-capture", tabId: selectedId });
  renderStatus();
  renderButtons();
});

$("stop").addEventListener("click", async () => {
  state = await call({ type: "stop-capture" });
  renderStatus();
  renderButtons();
});

$("refresh").addEventListener("click", async () => {
  await call({ type: "save-settings", port: $("port").value, token: $("token").value });
  $("token").value = "";
  await refresh();
});

$("save").addEventListener("click", async () => {
  const port = Number($("port").value);
  if (!port) return;
  state = await call({ type: "save-settings", port, token: $("token").value });
  $("token").value = "";
  renderStatus();
  renderButtons();
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "state") {
    state = msg.state;
    renderStatus();
    renderButtons();
  }
});

refresh();
timer = setInterval(refresh, 2000);
window.addEventListener("unload", () => clearInterval(timer));
