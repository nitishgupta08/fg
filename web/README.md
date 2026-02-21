# functiongemma web

Responsive single-page chat app with:
- model discovery from `/v1/models`
- smart-home tool-calling chat
- request/response logs

## Behavior
- On load, app immediately calls `GET /v1/models` from fixed base URL:
  - `http://127.0.0.1:8080/v1/models`
- Loaded models populate the chat model selector.
- If model loading fails, chat actions are disabled until retry succeeds.

## Logs
- Every `/v1/models` request/response is logged.
- Every model invocation request/response/error is logged.
- Use `Clear Logs` to reset the log stream.

## Run
```bash
cd web
npm install
npm run dev
```

Build static output:

```bash
npm run build
```

Preview build:

```bash
npm run preview
```

Run the built files:

```bash
cd dist/
python3 -m http.server 4173
```
