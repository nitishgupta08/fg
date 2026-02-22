"""Smart Home Controller — Text-based orchestrator.

Pairs a lightweight SLM client (OpenAI-compatible API) with a deterministic
orchestrator that handles slot elicitation, dialogue control, and simulated
backend execution for smart home control.

Usage:
    Interactive:
        uv run main.py --model functiongemma-270m-it-BF16 --port 8080
        uv run main.py --model distil-home-assistant-functiongemma --port 8080
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from openai import OpenAI

from action_executor import execute_action
# ---------------------------------------------------------------------------
# Tools definition (6 smart home functions)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "toggle_lights",
            "description": "Turn lights on or off in a specified room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "enum": [
                            "living_room",
                            "bedroom",
                            "kitchen",
                            "bathroom",
                            "office",
                            "hallway",
                        ],
                        "description": "The room whose lights to control.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Whether to turn lights on or off.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the temperature for heating or cooling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "integer",
                        "minimum": 60,
                        "maximum": 80,
                        "description": "The target temperature in degrees Fahrenheit (60-80).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["heat", "cool", "auto"],
                        "description": "The thermostat mode: heat, cool, or auto.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_door",
            "description": "Lock or unlock a door.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door": {
                        "type": "string",
                        "enum": ["front", "back", "garage", "side"],
                        "description": "Which door to lock or unlock.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["lock", "unlock"],
                        "description": "Whether to lock or unlock the door.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": "Query the current state of a device or room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "enum": ["lights", "thermostat", "door", "all"],
                        "description": "The type of device to check.",
                    },
                    "room": {
                        "type": "string",
                        "description": "The room or location to check.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scene",
            "description": "Activate a predefined scene that controls multiple devices at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {
                        "type": "string",
                        "enum": [
                            "movie_night",
                            "bedtime",
                            "morning",
                            "away",
                            "party",
                        ],
                        "description": "The scene to activate.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "intent_unclear",
            "description": "Call when the user's request is ambiguous, off-topic, or cannot be mapped to any available smart home function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": [
                            "ambiguous",
                            "off_topic",
                            "incomplete",
                            "unsupported_device",
                        ],
                        "description": "Why the intent is unclear.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Orchestrator constants
# ---------------------------------------------------------------------------
FUNCTION_REQUIRED_ARGS: dict[str, list[str]] = {
    "toggle_lights": ["room", "state"],
    "set_thermostat": ["temperature"],
    "lock_door": ["door", "state"],
    "get_device_status": ["device_type"],
    "set_scene": ["scene"],
    "intent_unclear": [],
}

INDIVIDUAL_SLOT_PROMPTS: dict[str, dict[str, str]] = {
    "toggle_lights": {
        "room": "which room (living room, bedroom, kitchen, bathroom, office, or hallway)",
        "state": "whether to turn them on or off",
    },
    "set_thermostat": {
        "temperature": "what temperature (60-80°F)",
        "mode": "the mode (heat, cool, or auto)",
    },
    "lock_door": {
        "door": "which door (front, back, garage, or side)",
        "state": "whether to lock or unlock it",
    },
    "get_device_status": {
        "device_type": "which device type (lights, thermostat, door, or all)",
        "room": "which room or location",
    },
    "set_scene": {
        "scene": "which scene (movie night, bedtime, morning, away, or party)",
    },
}

# ---------------------------------------------------------------------------
# SLM Client — stateless wrapper around an OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a tool-calling model working on:\n"
        "<task_description>You are an on-device smart home controller. "
        "Given a natural language command from the user, call the appropriate "
        "smart home function. If the user does not specify a required value "
        "(e.g. which room or what temperature), omit that parameter from the "
        "function call. Maintain context across conversation turns to resolve "
        "pronouns and sequential commands.</task_description>\n\n"
        "Respond to the conversation history by generating an appropriate tool call that "
        "satisfies the user request. Generate only the tool call according to the provided "
        "tool schema, do not generate anything else. Always respond with a tool call.\n\n"
    ),
}


class SLMClient:
    """Lightweight client for a llama.cpp / Ollama / vLLM server."""

    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 8000):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key=api_key,
        )

    def invoke(self, conversation_history: list[dict]) -> tuple[dict | str, float]:
        """Send full conversation history to the SLM and return parsed tool call + latency."""
        messages = [SYSTEM_PROMPT] + conversation_history

        start = time.perf_counter()
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            tools=TOOLS,
            tool_choice="required",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        response = chat_response.choices[0].message

        # Path A: proper tool_calls in the response.
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return {"name": fn.name, "arguments": arguments}, latency_ms

        # Path B: model returned JSON in content (fallback).
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    return {"name": parsed["name"], "arguments": args}, latency_ms
            except (json.JSONDecodeError, KeyError):
                pass

        return (
            f"No valid tool call in SLM response, model returned {response}",
            latency_ms,
        )


# ---------------------------------------------------------------------------
# Text Orchestrator
# ---------------------------------------------------------------------------
class TextOrchestrator:
    """Deterministic dialogue manager sitting between the user and the SLM."""

    def __init__(self, slm_client: SLMClient, debug: bool = False):
        self.slm = slm_client
        self.debug = debug
        self.conversation_history: list[dict] = []
        self.last_function_call: dict | str | None = None
        self.last_latency_ms: float = 0.0

    def process_utterance(self, transcript: str) -> str | None:
        """Full turn: user text in -> bot response out."""
        if transcript.lower() in ("quit", "exit"):
            return None

        # 1. Append user turn.
        self.conversation_history.append({"role": "user", "content": transcript})

        # 2. Call SLM.
        function_call, latency_ms = self.slm.invoke(self.conversation_history)
        self.last_function_call = function_call
        self.last_latency_ms = latency_ms

        if self.debug:
            print(f"  [DEBUG] SLM returned ({latency_ms:.1f} ms): {function_call}")

        # 3. If the SLM failed to return a valid call, treat as unclear.
        if isinstance(function_call, str):
            self.conversation_history.append({"role": "assistant", "content": ""})
            return self.generate_clarification_response()

        # 4. Record assistant turn in history (tool_calls format).
        args_str = (
            json.dumps(function_call["arguments"])
            if isinstance(function_call["arguments"], dict)
            else function_call["arguments"]
        )
        tool_call_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": args_str,
                    },
                }
            ],
        }
        self.conversation_history.append(tool_call_msg)

        # 5. Route through orchestrator logic.
        response = self.handle_function_call(function_call)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def reset(self) -> None:
        self.conversation_history = []
        self.last_function_call = None
        self.last_latency_ms = 0.0

    def handle_function_call(self, function_call: dict) -> str:
        name = function_call["name"]
        arguments = function_call.get("arguments", {})

        if name == "intent_unclear":
            return self.generate_clarification_response()

        missing = self.get_missing_args(name, arguments)
        if missing:
            return self.generate_slot_elicitation(name, missing, arguments)

        return self.execute_and_respond(name, arguments)

    def get_missing_args(self, function_name: str, arguments: dict) -> list[str]:
        required = FUNCTION_REQUIRED_ARGS.get(function_name, [])
        return [arg for arg in required if arguments.get(arg) is None]

    def generate_clarification_response(self) -> str:
        capabilities = [
            "control lights",
            "set the thermostat",
            "lock or unlock doors",
            "check device status",
            "or activate scenes",
        ]
        return (
            "I didn't quite understand that. Could you tell me what you need? "
            f"I can help you {', '.join(capabilities)}."
        )

    def generate_slot_elicitation(
        self, function: str, missing_args: list[str], current_args: dict
    ) -> str:
        _ = current_args
        individual = INDIVIDUAL_SLOT_PROMPTS.get(function, {})
        questions = [
            individual.get(arg, f"the {arg.replace('_', ' ')}") for arg in missing_args
        ]
        if len(questions) == 1:
            return f"Could you provide {questions[0]}?"
        return f"Could you provide {', '.join(questions[:-1])}, and {questions[-1]}?"

    def execute_and_respond(self, function: str, arguments: dict[str, Any]) -> str:
        result = execute_action(
            action=function,
            arguments=arguments,
            debug=self.debug,
        )

        if result.ok:
            return (
                f"Done. Action '{function}' accepted (status {result.status_code}). "
                f"Message: {result.message} "
                f"[request_id={result.request_id}]"
            )

        return (
            f"Action failed for '{function}'. "
            f"Status: {result.status_code if result.status_code is not None else 'n/a'}. "
            f"Message: {result.message} "
            f"[request_id={result.request_id}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Home Controller")
    parser.add_argument(
        "--model",
        type=str,
        default="functiongemma-270m-it",
        help="Model name served by llama-server",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port of the OpenAI-compatible server"
    )
    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="API key (default EMPTY)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print raw SLM output each turn"
    )
    args = parser.parse_args()

    slm = SLMClient(model_name=args.model, api_key=args.api_key, port=args.port)
    orchestrator = TextOrchestrator(slm, debug=args.debug)

    print("Smart Home Controller (type 'quit' or 'exit' to stop)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = orchestrator.process_utterance(user_input)
            if response is None:
                print("Bot: Goodbye!")
                break
            print(f"Bot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBot: Goodbye!")


if __name__ == "__main__":
    main()
