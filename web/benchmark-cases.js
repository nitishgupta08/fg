export const BENCHMARK_CASES = [
  {
    name: "lights_simple",
    dialogue: ["Turn on the kitchen lights."],
    expected: { name: "toggle_lights", arguments: { room: "kitchen", state: "on" } },
  },
  {
    name: "thermostat_basic",
    dialogue: ["Set thermostat to 70 degrees."],
    expected: { name: "set_thermostat", arguments: { temperature: 70 } },
  },
  {
    name: "door_lock",
    dialogue: ["Lock the garage door."],
    expected: { name: "lock_door", arguments: { door: "garage", state: "lock" } },
  },
  {
    name: "scene",
    dialogue: ["Activate bedtime scene."],
    expected: { name: "set_scene", arguments: { scene: "bedtime" } },
  },
  {
    name: "status",
    dialogue: ["What is the thermostat status?"],
    expected: { name: "get_device_status", arguments: { device_type: "thermostat" } },
  },
  {
    name: "contextual_followup",
    dialogue: ["Turn on lights in the office.", "Now switch them off."],
    expected: { name: "toggle_lights", arguments: { room: "office", state: "off" } },
  },
];
