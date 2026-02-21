import { BENCHMARK_CASES } from "./benchmark-cases.js";
import {
  createOrchestratorState,
  downloadJsonReport,
  processUtterance,
  resetOrchestratorState,
  runBenchmark,
} from "./orchestrator.js";

const API_BASE_URL = "http://127.0.0.1:8080/v1";
const API_KEY = "EMPTY";

const elements = {
  menuToggle: document.getElementById("menu-toggle"),
  mainMenu: document.getElementById("main-menu"),
  viewButtons: Array.from(document.querySelectorAll("[data-view-target]")),
  views: Array.from(document.querySelectorAll("[data-view]")),

  modelsStatus: document.getElementById("models-status"),
  reloadModels: document.getElementById("reload-models"),

  chatLog: document.getElementById("chat-log"),
  chatModel: document.getElementById("chat-model"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatReset: document.getElementById("chat-reset"),

  benchBaseModel: document.getElementById("bench-base-model"),
  benchDistilModel: document.getElementById("bench-distil-model"),
  benchRun: document.getElementById("run-benchmark"),
  benchDownload: document.getElementById("download-report"),
  benchSummary: document.getElementById("benchmark-summary"),
  benchTableBody: document.getElementById("benchmark-table-body"),
  benchStatus: document.getElementById("benchmark-status"),

  logs: document.getElementById("logs"),
  clearLogs: document.getElementById("clear-logs"),
};

const chatState = createOrchestratorState("");
let latestReport = null;
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
  elements.benchBaseModel.disabled = !enabled;
  elements.benchDistilModel.disabled = !enabled;
  elements.benchRun.disabled = !enabled;
}

function renderModelOptions(models) {
  const selects = [elements.chatModel, elements.benchBaseModel, elements.benchDistilModel];

  selects.forEach((select) => {
    select.innerHTML = "";
    models.forEach((modelId) => {
      const option = document.createElement("option");
      option.value = modelId;
      option.textContent = modelId;
      select.append(option);
    });
  });

  if (models.length > 0) {
    const distilCandidate =
      models.find((id) => id.toLowerCase().includes("distil")) || models[models.length - 1];
    const baseCandidate =
      models.find((id) => !id.toLowerCase().includes("distil")) || models[0];

    elements.chatModel.value = baseCandidate;
    elements.benchBaseModel.value = baseCandidate;
    elements.benchDistilModel.value = distilCandidate;
    resetOrchestratorState(chatState, baseCandidate);
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

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMs(value) {
  return `${value.toFixed(1)} ms`;
}

function buildCaseLookup(modelCases) {
  const map = new Map();
  modelCases.forEach((result) => map.set(result.name, result));
  return map;
}

function renderBenchmarkReport(report, config) {
  const base = report.models.base;
  const distil = report.models.distil;

  elements.benchSummary.textContent = [
    `Base (${config.baseModel}): accuracy ${formatPercent(base.summary.accuracy)}, avg latency ${formatMs(base.summary.avgLatencyMs)}, passed ${base.summary.passedCases}/${base.summary.totalCases}`,
    `Distil (${config.distilModel}): accuracy ${formatPercent(distil.summary.accuracy)}, avg latency ${formatMs(distil.summary.avgLatencyMs)}, passed ${distil.summary.passedCases}/${distil.summary.totalCases}`,
    `Delta (distil - base): pass-count ${report.comparison.passCountDelta}, accuracy ${(report.comparison.accuracyDelta * 100).toFixed(1)}pp, avg latency ${report.comparison.avgLatencyDeltaMs.toFixed(1)} ms`,
  ].join("\n");

  elements.benchTableBody.innerHTML = "";
  const baseLookup = buildCaseLookup(base.cases);
  const distilLookup = buildCaseLookup(distil.cases);

  for (const benchCase of BENCHMARK_CASES) {
    const baseCase = baseLookup.get(benchCase.name);
    const distilCase = distilLookup.get(benchCase.name);

    const row = document.createElement("tr");

    const cells = [
      ["Case", benchCase.name],
      ["Expected", benchCase.expected.name],
      ["Base Tool", baseCase?.actual?.name || "(none)"],
      ["Base", baseCase?.passed ? "pass" : "fail"],
      ["Base Lat", baseCase ? `${baseCase.latencyMs.toFixed(1)} ms` : "0.0 ms"],
      ["Distil Tool", distilCase?.actual?.name || "(none)"],
      ["Distil", distilCase?.passed ? "pass" : "fail"],
      ["Distil Lat", distilCase ? `${distilCase.latencyMs.toFixed(1)} ms` : "0.0 ms"],
    ];

    cells.forEach(([label, value]) => {
      const cell = document.createElement("td");
      cell.setAttribute("data-label", label);
      cell.textContent = value;
      row.append(cell);
    });

    elements.benchTableBody.append(row);
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

async function onRunBenchmark() {
  if (elements.benchRun.disabled) return;

  const benchConfig = {
    baseUrl: API_BASE_URL,
    apiKey: API_KEY,
    baseModel: elements.benchBaseModel.value,
    distilModel: elements.benchDistilModel.value,
  };

  elements.benchStatus.textContent = "Running benchmark...";
  elements.benchRun.disabled = true;

  try {
    const report = await runBenchmark(benchConfig, BENCHMARK_CASES, { onLog: handleModelLog });
    latestReport = report;
    elements.benchDownload.disabled = false;
    renderBenchmarkReport(report, benchConfig);
    elements.benchStatus.textContent = `Completed at ${new Date(report.generatedAt).toLocaleString()}`;
  } catch (err) {
    addLog("error", "Benchmark run failed", { message: String(err) });
    elements.benchStatus.textContent = `Benchmark failed: ${String(err)}`;
  } finally {
    elements.benchRun.disabled = elements.benchBaseModel.disabled;
  }
}

async function onChatSend() {
  const text = elements.chatInput.value.trim();
  if (!text || elements.chatSend.disabled) return;

  const model = elements.chatModel.value;

  appendChat("user", text);
  elements.chatInput.value = "";
  elements.chatSend.disabled = true;

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

    appendChat("assistant", response.assistantText);
  } catch (err) {
    addLog("error", "Chat request failed", { message: String(err) });
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

function switchView(viewName) {
  elements.views.forEach((view) => {
    view.classList.toggle("active", view.dataset.view === viewName);
  });

  elements.viewButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.viewTarget === viewName);
  });

  elements.mainMenu.hidden = true;
  elements.menuToggle.setAttribute("aria-expanded", "false");
}

function toggleMenu() {
  const isHidden = elements.mainMenu.hidden;
  elements.mainMenu.hidden = !isHidden;
  elements.menuToggle.setAttribute("aria-expanded", String(isHidden));
}

function init() {
  elements.menuToggle.addEventListener("click", toggleMenu);
  elements.viewButtons.forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.viewTarget));
  });

  elements.reloadModels.addEventListener("click", loadModelsFlow);
  elements.chatSend.addEventListener("click", onChatSend);
  elements.chatReset.addEventListener("click", onChatReset);
  elements.benchRun.addEventListener("click", onRunBenchmark);
  elements.benchDownload.addEventListener("click", () => {
    if (latestReport) downloadJsonReport(latestReport);
  });

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

  switchView("chat");
  loadModelsFlow();
}

init();
