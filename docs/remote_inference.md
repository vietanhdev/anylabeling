# Authenticated remote inference

AnyLabeling can use a centrally hosted AnyLearning ONNX model without copying
the model to every labeling computer. The desktop client sends the encoded
image and bounded inference metadata; the server returns editable shapes.

## Configure the server

Follow the [AnyLearning server guide](https://github.com/nrl-ai/anylearning-oss/blob/develop/docs/server.md)
to configure the inference service and its startup model manifest. The server
chooses every model path and backend setting; clients cannot upload models.

Use direct TLS or a trusted TLS-terminating reverse proxy for every network
deployment. Plain HTTP is accepted only for localhost and numeric loopback
addresses.

## Configure AnyLabeling

Keep the plaintext password out of YAML. Put it in the environment before
launching AnyLabeling:

```shell
export ANYLABELING_REMOTE_PASSWORD='use-a-long-random-password'
anylabeling
```

On PowerShell:

```powershell
$env:ANYLABELING_REMOTE_PASSWORD = 'use-a-long-random-password'
anylabeling
```

Create a custom model YAML file and select it in the auto-labeling model picker:

```yaml
type: remote
name: shared-yolox
display_name: Shared YOLOX detector
server_url: https://inference.example.com
model_id: shared-detector
password_env: ANYLABELING_REMOTE_PASSWORD
prediction_timeout_seconds: 120
poll_interval_seconds: 0.1
parameters:
  confidence: 0.5
  iou: 0.45
```

`model_id` must match the server's immutable startup manifest. `parameters` are
optional bounded values; the selected AnyLearning backend decides which names
it supports.

The client exchanges the password for a short-lived token and keeps both in
memory only. It ignores proxy environment variables, rejects redirects,
verifies HTTPS certificates, hashes the exact encoded image into the request
identity, bounds responses, and deletes completed or cancelled jobs. Never put
passwords or tokens in YAML, URLs, command lines, screenshots, or issue reports.

Interactive models expose point and rectangle controls when the server
advertises `promptable_segmentation`. Detection models expose the Run button.

## Troubleshooting

- `non-loopback remote inference requires HTTPS`: use HTTPS, or test locally
  through `http://127.0.0.1:<port>`.
- `Remote authentication failed`: ensure the configured environment variable
  exists in the process that launched AnyLabeling and matches the server hash.
- `Server token lifetime is shorter than the prediction timeout`: raise the
  server token TTL or lower `prediction_timeout_seconds`.
- `Remote inference capacity was reached`: the bounded server queue is full;
  wait for current jobs or deliberately raise measured server limits.
