# functiongemma web

Responsive single-page app with hamburger navigation:
- Chat page
- Benchmark page

## Behavior
- No connection config UI.
- On load, app immediately calls `GET /v1/models` from fixed base URL:
  - `http://127.0.0.1:8080/v1/models`
- Loaded models populate:
  - chat model selector
  - benchmark base/distil selectors
- If model loading fails, chat and benchmark actions are disabled until retry succeeds.

## Logs
Diagnostics is replaced with request/response logs.
- Every `/v1/models` request/response is logged.
- Every model invocation request/response/error is logged.
- Use `Clear Logs` to reset the log stream.

## Run
From `/Users/nitishgupta/Developer/functiongemma/web`:

```bash
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
