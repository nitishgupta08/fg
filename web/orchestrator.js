import OpenAI from "openai";
import {
  FUNCTION_REQUIRED_ARGS,
  INDIVIDUAL_SLOT_PROMPTS,
  ROOM_DISPLAY,
  SCENE_DESCRIPTIONS,
  SUCCESS_TEMPLATES,
  SYSTEM_PROMPT,
  TOOLS,
} from "./tools.js";

let openAIClientState = {
  key: null,
  client: null,
};

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function formatTemplate(template, values) {
  return template.replaceAll(/\{([^}]+)\}/g, (_, key) => {
    const value = values[key];
    return value == null ? "" : String(value);
  });
}

function deterministicPick(seed, values) {
  const text = String(seed || "default");
  let total = 0;
  for (let i = 0; i < text.length; i += 1) {
    total += text.charCodeAt(i);
  }
  return values[total % values.length];
}

function normalizeBaseUrl(baseUrl) {
  return (baseUrl || "").replace(/\/+$/, "");
}

function getOpenAIClient(config) {
  const key = `${config.baseUrl}|${config.apiKey || "EMPTY"}`;
  if (openAIClientState.key === key && openAIClientState.client) {
    return openAIClientState.client;
  }

  const client = new OpenAI({
    apiKey: config.apiKey || "EMPTY",
    baseURL: normalizeBaseUrl(config.baseUrl),
    dangerouslyAllowBrowser: true,
  });
  openAIClientState = { key, client };
  return client;
}

function parseToolCallFromMessage(message) {
  if (message?.tool_calls?.length) {
    const fn = message.tool_calls[0].function;
    let args = fn.arguments || {};
    if (typeof args === "string") {
      args = JSON.parse(args);
    }
    return { name: fn.name, arguments: args };
  }

  const content = message?.content;
  if (content && typeof content === "string") {
    const parsed = JSON.parse(content.trim());
    if (parsed?.name) {
      let args = parsed.arguments ?? parsed.parameters ?? {};
      if (typeof args === "string") {
        args = JSON.parse(args);
      }
      return { name: parsed.name, arguments: args };
    }
  }

  throw new Error("No valid tool call in model response.");
}

async function invokeViaSdk(payload, config) {
  const client = getOpenAIClient(config);

  const response = await client.chat.completions.create(payload);
  const message = response?.choices?.[0]?.message;
  const toolCall = parseToolCallFromMessage(message);

  return {
    toolCall,
    raw: response,
    transport: "openai-sdk",
  };
}

export async function invokeModel({
  baseUrl,
  apiKey,
  model,
  conversationHistory,
  tools = TOOLS,
  onLog,
}) {
  const payload = {
    model,
    messages: [SYSTEM_PROMPT, ...conversationHistory],
    temperature: 0,
    tools,
    tool_choice: "required",
    extra_body: { chat_template_kwargs: { enable_thinking: false } },
  };

  const config = { baseUrl, apiKey };
  const started = performance.now();

  try {
    const endpoint = `${normalizeBaseUrl(baseUrl)}/chat/completions`;
    if (typeof onLog === "function") {
      onLog({
        type: "request",
        endpoint,
        model,
        payload,
      });
    }

    const sdkResult = await invokeViaSdk(payload, config);
    if (typeof onLog === "function") {
      onLog({
        type: "response",
        endpoint,
        model,
        latencyMs: performance.now() - started,
        transport: sdkResult.transport,
        toolCall: sdkResult.toolCall,
      });
    }
    return {
      toolCall: sdkResult.toolCall,
      latencyMs: performance.now() - started,
      raw: sdkResult.raw,
      transport: sdkResult.transport,
      error: null,
    };
  } catch (err) {
    const endpoint = `${normalizeBaseUrl(baseUrl)}/chat/completions`;
    if (typeof onLog === "function") {
      onLog({
        type: "error",
        endpoint,
        model,
        error: String(err),
      });
    }
    return {
      toolCall: null,
      latencyMs: performance.now() - started,
      raw: null,
      transport: "failed",
      error: `OpenAI SDK error: ${String(err)}`,
    };
  }
}

export function createOrchestratorState(model = "") {
  return {
    model,
    conversationHistory: [],
    lastFunctionCall: null,
    lastLatencyMs: 0,
    lastRaw: null,
    lastError: null,
    lastTransport: null,
  };
}

export function resetOrchestratorState(state, model = state.model) {
  state.model = model;
  state.conversationHistory = [];
  state.lastFunctionCall = null;
  state.lastLatencyMs = 0;
  state.lastRaw = null;
  state.lastError = null;
  state.lastTransport = null;
}

function getMissingArgs(functionName, argumentsMap) {
  const required = FUNCTION_REQUIRED_ARGS[functionName] || [];
  return required.filter((arg) => argumentsMap?.[arg] == null);
}

function generateClarificationResponse() {
  return (
    "I didn't quite understand that. Could you tell me what you need? " +
    "I can help you control lights, set the thermostat, lock or unlock doors, check device status, or activate scenes."
  );
}

function generateSlotElicitation(functionName, missingArgs) {
  const individual = INDIVIDUAL_SLOT_PROMPTS[functionName] || {};
  const questions = missingArgs.map((arg) => individual[arg] || `the ${arg.replaceAll("_", " ")}`);
  if (questions.length === 1) {
    return `Could you provide ${questions[0]}?`;
  }
  return `Could you provide ${questions.slice(0, -1).join(", ")}, and ${questions[questions.length - 1]}?`;
}

function simulateDeviceStatus(argumentsMap) {
  const deviceType = argumentsMap.device_type || "all";
  const room = argumentsMap.room || "";

  if (deviceType === "lights") {
    const state = deterministicPick(`${room}|lights`, ["on", "off"]);
    const display = room ? ROOM_DISPLAY[room] || room : "home";
    return `The ${display} lights are currently ${state}.`;
  }

  if (deviceType === "thermostat") {
    const temp = deterministicPick("thermostat-temp", [68, 69, 70, 71, 72]);
    const mode = deterministicPick("thermostat-mode", ["heat", "cool", "auto"]);
    return `The thermostat is set to ${temp}°F in ${mode} mode.`;
  }

  if (deviceType === "door") {
    const door = room || "front";
    const state = deterministicPick(`${door}|door`, ["locked", "unlocked"]);
    return `The ${door} door is currently ${state}.`;
  }

  return "Lights: mixed (some on, some off). Thermostat: 70°F. Doors: front locked, back locked, garage unlocked.";
}

function callBackendApi(functionName, argumentsMap) {
  if (functionName === "toggle_lights") {
    const room = argumentsMap.room || "";
    return { display_room: ROOM_DISPLAY[room] || room };
  }

  if (functionName === "set_thermostat") {
    const mode = argumentsMap.mode;
    return { mode_suffix: mode ? ` in ${mode} mode` : "" };
  }

  if (functionName === "get_device_status") {
    return { status_report: simulateDeviceStatus(argumentsMap) };
  }

  if (functionName === "set_scene") {
    const scene = argumentsMap.scene || "";
    return { scene_details: SCENE_DESCRIPTIONS[scene] || "" };
  }

  return {};
}

function executeAndRespond(functionName, argumentsMap) {
  const apiResult = callBackendApi(functionName, argumentsMap);
  const template = SUCCESS_TEMPLATES[functionName] || "Done.";
  return formatTemplate(template, { ...argumentsMap, ...apiResult });
}

function handleFunctionCall(functionCall) {
  const name = functionCall.name;
  const argumentsMap = functionCall.arguments || {};

  if (name === "intent_unclear") {
    return generateClarificationResponse();
  }

  const missing = getMissingArgs(name, argumentsMap);
  if (missing.length) {
    return generateSlotElicitation(name, missing);
  }

  return executeAndRespond(name, argumentsMap);
}

function toAssistantToolCallMessage(functionCall) {
  return {
    role: "assistant",
    tool_calls: [
      {
        type: "function",
        function: {
          name: functionCall.name,
          arguments: JSON.stringify(functionCall.arguments || {}),
        },
      },
    ],
  };
}

export async function processUtterance(state, text, config, options = {}) {
  const input = String(text || "").trim();
  if (!input) {
    return {
      assistantText: "Please enter a message.",
      functionCall: null,
      latencyMs: 0,
      error: "empty_input",
      raw: null,
    };
  }

  state.conversationHistory.push({ role: "user", content: input });

  const modelResponse = await invokeModel({
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    model: config.model || state.model,
    conversationHistory: state.conversationHistory,
    tools: TOOLS,
    onLog: options.onLog,
  });

  state.lastLatencyMs = modelResponse.latencyMs;
  state.lastRaw = modelResponse.raw;
  state.lastError = modelResponse.error;
  state.lastTransport = modelResponse.transport;

  if (!modelResponse.toolCall) {
    state.lastFunctionCall = null;
    const assistantText =
      "Model call failed. Check diagnostics and server CORS/connectivity settings, then try again.";
    state.conversationHistory.push({ role: "assistant", content: assistantText });
    return {
      assistantText,
      functionCall: null,
      latencyMs: modelResponse.latencyMs,
      error: modelResponse.error,
      raw: modelResponse.raw,
    };
  }

  state.lastFunctionCall = deepClone(modelResponse.toolCall);
  state.conversationHistory.push(toAssistantToolCallMessage(modelResponse.toolCall));

  const assistantText = handleFunctionCall(modelResponse.toolCall);
  state.conversationHistory.push({ role: "assistant", content: assistantText });

  return {
    assistantText,
    functionCall: modelResponse.toolCall,
    latencyMs: modelResponse.latencyMs,
    error: null,
    raw: modelResponse.raw,
  };
}
