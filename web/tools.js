export const TOOLS = [
  {
    type: "function",
    function: {
      name: "toggle_lights",
      description: "Turn lights on or off in a specified room.",
      parameters: {
        type: "object",
        properties: {
          room: {
            type: "string",
            enum: ["living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"],
            description: "The room whose lights to control.",
          },
          state: {
            type: "string",
            enum: ["on", "off"],
            description: "Whether to turn lights on or off.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_thermostat",
      description: "Set the temperature for heating or cooling.",
      parameters: {
        type: "object",
        properties: {
          temperature: {
            type: "integer",
            minimum: 60,
            maximum: 80,
            description: "The target temperature in degrees Fahrenheit (60-80).",
          },
          mode: {
            type: "string",
            enum: ["heat", "cool", "auto"],
            description: "The thermostat mode: heat, cool, or auto.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "lock_door",
      description: "Lock or unlock a door.",
      parameters: {
        type: "object",
        properties: {
          door: {
            type: "string",
            enum: ["front", "back", "garage", "side"],
            description: "Which door to lock or unlock.",
          },
          state: {
            type: "string",
            enum: ["lock", "unlock"],
            description: "Whether to lock or unlock the door.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_device_status",
      description: "Query the current state of a device or room.",
      parameters: {
        type: "object",
        properties: {
          device_type: {
            type: "string",
            enum: ["lights", "thermostat", "door", "all"],
            description: "The type of device to check.",
          },
          room: {
            type: "string",
            description: "The room or location to check.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_scene",
      description: "Activate a predefined scene that controls multiple devices at once.",
      parameters: {
        type: "object",
        properties: {
          scene: {
            type: "string",
            enum: ["movie_night", "bedtime", "morning", "away", "party"],
            description: "The scene to activate.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "intent_unclear",
      description:
        "Call when the user's request is ambiguous, off-topic, or cannot be mapped to any available smart home function.",
      parameters: {
        type: "object",
        properties: {
          reason: {
            type: "string",
            enum: ["ambiguous", "off_topic", "incomplete", "unsupported_device"],
            description: "Why the intent is unclear.",
          },
        },
        required: [],
        additionalProperties: false,
      },
    },
  },
];

export const SYSTEM_PROMPT = {
  role: "system",
  content:
    "You are a tool-calling model working on:\n" +
    "<task_description>You are an on-device smart home controller. Given a natural language command from the user, call the appropriate smart home function. " +
    "If the user does not specify a required value (e.g. which room or what temperature), omit that parameter from the function call. " +
    "Maintain context across conversation turns to resolve pronouns and sequential commands.</task_description>\n\n" +
    "Respond to the conversation history by generating an appropriate tool call that satisfies the user request. " +
    "Generate only the tool call according to the provided tool schema, do not generate anything else. " +
    "Always respond with a tool call.\n\n",
};

export const FUNCTION_REQUIRED_ARGS = {
  toggle_lights: ["room", "state"],
  set_thermostat: ["temperature"],
  lock_door: ["door", "state"],
  get_device_status: ["device_type"],
  set_scene: ["scene"],
  intent_unclear: [],
};

export const INDIVIDUAL_SLOT_PROMPTS = {
  toggle_lights: {
    room: "which room (living room, bedroom, kitchen, bathroom, office, or hallway)",
    state: "whether to turn them on or off",
  },
  set_thermostat: {
    temperature: "what temperature (60-80°F)",
    mode: "the mode (heat, cool, or auto)",
  },
  lock_door: {
    door: "which door (front, back, garage, or side)",
    state: "whether to lock or unlock it",
  },
  get_device_status: {
    device_type: "which device type (lights, thermostat, door, or all)",
    room: "which room or location",
  },
  set_scene: {
    scene: "which scene (movie night, bedtime, morning, away, or party)",
  },
};

export const SCENE_DESCRIPTIONS = {
  movie_night: "Living room lights dimmed, thermostat set to 72°F.",
  bedtime: "All lights off, doors locked, thermostat set to 68°F.",
  morning: "Kitchen and hallway lights on, thermostat set to 72°F.",
  away: "All lights off, all doors locked, thermostat set to 65°F.",
  party: "Living room and kitchen lights on, thermostat set to 70°F.",
};

export const ROOM_DISPLAY = {
  living_room: "living room",
  bedroom: "bedroom",
  kitchen: "kitchen",
  bathroom: "bathroom",
  office: "office",
  hallway: "hallway",
};

export const SUCCESS_TEMPLATES = {
  toggle_lights: "Done. The {display_room} lights are now {state}.",
  set_thermostat: "Done. Thermostat set to {temperature}°F{mode_suffix}.",
  lock_door: "Done. The {door} door is now {state}ed.",
  get_device_status: "{status_report}",
  set_scene: 'Done. "{scene}" scene activated. {scene_details}',
  intent_unclear: "",
};
