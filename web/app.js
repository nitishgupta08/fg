import {
  createOrchestratorState,
  processUtterance,
  resetOrchestratorState,
} from "./orchestrator.js";

const API_BASE_URL = "http://127.0.0.1:8080/v1";
const API_KEY = "EMPTY";

const elements = {
  modelsStatus: document.getElementById("models-status"),
  reloadModels: document.getElementById("reload-models"),

  chatLog: document.getElementById("chat-log"),
  chatModel: document.getElementById("chat-model"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatReset: document.getElementById("chat-reset"),

  logs: document.getElementById("logs"),
  clearLogs: document.getElementById("clear-logs"),
};

const chatState = createOrchestratorState("");
let modelIds = [];
let logEntries = [];

function addLog(type, title, payload) {
  const entry = {
    timestamp: new Date().toISOString(),
    type,
    title,
    payload,
  };
  logEntries.unshift(entry);
  logEntries = logEntries.slice(0, 250);
  renderLogs();
}

function renderLogs() {
  elements.logs.innerHTML = "";

  for (const entry of logEntries) {
    const item = document.createElement("article");
    item.className = `log-item ${entry.type}`;

    const heading = document.createElement("div");
    heading.className = "log-heading";
    heading.textContent = `[${entry.type.toUpperCase()}] ${entry.title} @ ${entry.timestamp}`;

    const body = document.createElement("pre");
    body.className = "log-body";
    body.textContent = JSON.stringify(entry.payload, null, 2);

    item.append(heading, body);
    elements.logs.append(item);
  }
}

function appendChat(role, text) {
  const item = document.createElement("div");
  item.className = `chat-item ${role}`;
  item.textContent = `${role === "user" ? "You" : "Bot"}: ${text}`;
  elements.chatLog.append(item);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function setModelsLoadState({ loading, error, hasModels }) {
  elements.reloadModels.disabled = loading;

  if (loading) {
    elements.modelsStatus.textContent = "Loading models from /v1/models...";
  } else if (error) {
    elements.modelsStatus.textContent = `Failed to load models: ${error}`;
  } else if (hasModels) {
    elements.modelsStatus.textContent = `Loaded ${modelIds.length} model(s).`;
  } else {
    elements.modelsStatus.textContent = "No models found.";
  }

  const enabled = !loading && !error && hasModels;
  elements.chatModel.disabled = !enabled;
  elements.chatSend.disabled = !enabled;
}

function renderModelOptions(models) {
  elements.chatModel.innerHTML = "";

  models.forEach((modelId) => {
    const option = document.createElement("option");
    option.value = modelId;
    option.textContent = modelId;
    elements.chatModel.append(option);
  });

  if (models.length > 0) {
    elements.chatModel.value = models[0];
    resetOrchestratorState(chatState, models[0]);
  }
}

async function fetchModels() {
  const endpoint = `${API_BASE_URL}/models`;
  addLog("request", "Fetch models", { method: "GET", endpoint });

  const response = await fetch(endpoint, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
    },
  });

  if (!response.ok) {
    const text = await response.text();
    addLog("response", "Fetch models failed", {
      endpoint,
      status: response.status,
      body: text,
    });
    throw new Error(`HTTP ${response.status}`);
  }

  const payload = await response.json();
  const models = Array.isArray(payload?.data)
    ? payload.data.map((item) => item?.id).filter((id) => typeof id === "string")
    : [];

  const uniqueSorted = [...new Set(models)].sort((a, b) => a.localeCompare(b));

  addLog("response", "Fetch models success", {
    endpoint,
    status: response.status,
    modelCount: uniqueSorted.length,
    models: uniqueSorted,
  });

  return uniqueSorted;
}

async function loadModelsFlow() {
  modelIds = [];
  renderModelOptions([]);
  setModelsLoadState({ loading: true, error: null, hasModels: false });

  try {
    const models = await fetchModels();
    modelIds = models;
    renderModelOptions(models);

    if (!models.length) {
      setModelsLoadState({ loading: false, error: "empty model list", hasModels: false });
      return;
    }

    setModelsLoadState({ loading: false, error: null, hasModels: true });
  } catch (err) {
    addLog("error", "Model loading error", { message: String(err) });
    setModelsLoadState({ loading: false, error: String(err), hasModels: false });
  }
}

function handleModelLog(entry) {
  if (entry.type === "request") {
    addLog("request", "Model invoke request", entry);
  } else if (entry.type === "response") {
    addLog("response", "Model invoke response", entry);
  } else {
    addLog("error", "Model invoke error", entry);
  }
}

async function onChatSend() {
  const text = elements.chatInput.value.trim();
  if (!text || elements.chatSend.disabled) return;

  const model = elements.chatModel.value;

  appendChat("user", text);
  elements.chatInput.value = "";
  elements.chatSend.disabled = true;
  const turnStart = performance.now();

  try {
    const response = await processUtterance(
      chatState,
      text,
      {
        baseUrl: API_BASE_URL,
        apiKey: API_KEY,
        model,
      },
      { onLog: handleModelLog },
    );
    const turnMs = performance.now() - turnStart;

    appendChat("assistant", `${response.assistantText} (${turnMs.toFixed(1)} ms)`);
    addLog("response", "Chat turn completed", {
      model,
      turnMs: Number(turnMs.toFixed(1)),
      assistantText: response.assistantText,
      status: "ok",
    });
  } catch (err) {
    const turnMs = performance.now() - turnStart;
    addLog("error", "Chat request failed", { message: String(err) });
    addLog("response", "Chat turn completed", {
      model,
      turnMs: Number(turnMs.toFixed(1)),
      status: "error",
      error: String(err),
    });
    appendChat("assistant", `Error: ${String(err)}`);
  } finally {
    elements.chatSend.disabled = elements.chatModel.disabled;
  }
}

function onChatReset() {
  resetOrchestratorState(chatState, elements.chatModel.value || "");
  elements.chatLog.innerHTML = "";
  addLog("response", "Chat reset", { model: elements.chatModel.value || null });
}

function init() {
  elements.reloadModels.addEventListener("click", loadModelsFlow);
  elements.chatSend.addEventListener("click", onChatSend);
  elements.chatReset.addEventListener("click", onChatReset);

  elements.clearLogs.addEventListener("click", () => {
    logEntries = [];
    renderLogs();
  });

  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      onChatSend();
    }
  });

  loadModelsFlow();
}

init();
