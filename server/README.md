# Plaud Remote

A small self-hosted server that the Plaud Local app can send recordings to for
transcription and summarization with larger models than an iPhone can run.

It receives audio from the app, queues it, calls the model endpoints you
configure, and hands the transcript and summary back to the app. It runs no
models itself.

It is independent of the iOS code in this repository: one Docker container,
Python standard library only, no packages to install.

## Run it

```bash
cd server
PLAUD_REMOTE_ADMIN_PASSWORD='choose-a-password' docker compose up -d --build
```

Then open `http://<host>:8080` and sign in as `admin` with that password.
If you don't set a password, one is generated on first start and printed once
in the container log (`docker compose logs plaud-remote`).

The app only connects over **HTTPS**, so the server needs TLS before the app
can use it; see [HTTPS](#https) below.

Without Docker: `PLAUD_REMOTE_DATA_DIR=./data python -m plaud_remote` (Python 3.11+).

## Set it up

1. **Transcription**: the base URL and model of an OpenAI-compatible speech
   endpoint. The server posts each recording to `<base URL>/audio/transcriptions`.
   Examples: [Speaches](https://github.com/speaches-ai/speaches) or another
   faster-whisper server (`http://speaches:8000/v1`), or a hosted API.
   For **transcripts with speakers**, set **Endpoint type** to *NeMo-Speech.cpp*
   instead; see [Speaker transcripts](#speaker-transcripts).
2. **Summarization** (optional): the base URL and model of an OpenAI-compatible
   chat endpoint, called at `<base URL>/chat/completions`. Examples: Ollama
   (`http://ollama:11434/v1`), vLLM, LM Studio, or a hosted API. Leave it empty
   to return transcripts only; the app then summarizes on-device.
3. Use **Test connection** on each, then copy the **server address** and
   **API token** from the top of the page into the app: Settings → Processing →
   Remote server. The address must be the `https://` one.

Model endpoints must be reachable from inside the container. For a service on
the Docker host, use `http://host.docker.internal:<port>/v1` (on Linux, add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the compose file).

## Speaker transcripts

With **Endpoint type** set to *NeMo-Speech.cpp, Nemotron 3.5 Transcribe*, the server sends each
recording to NVIDIA's [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) serving
[Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF)
and gets back a punctuated transcript with speakers and timestamps. It uses NeMo-Speech.cpp's
`/chat/completions` route, because its `/audio/transcriptions` route returns lowercase text
without punctuation when speakers are requested.

- Run NeMo-Speech.cpp with the NVFP4 GGUF on a Blackwell GPU, or the IQ4_NL GGUF elsewhere,
  with the `…-Encoder-F16.gguf` file in the same folder: it holds the speaker and timestamp parts. For
  example, `nvcr.io/nvidia/nemo-speech.cpp:26.10 serve --asr-model <NVFP4 .gguf> --host 0.0.0.0
  --port 9000 --read-timeout 600 --write-timeout 1800`. The default 30-second socket timeouts are
  too short for long recordings. Set the base URL to `http://<host>:9000/v1` and the model to `salm`.
- NeMo-Speech.cpp can't read Ogg/Opus, so the server converts each recording to FLAC with
  `ffmpeg`. It adds about 400 MB to the image, so it's opt-in: build with
  `PLAUD_REMOTE_WITH_FFMPEG=1 docker compose up -d --build`. Without Docker, install ffmpeg
  yourself.
- Up to one hour per recording. Longer ones fail with a clear message.
- *Language* and *Prompt* are not used by this endpoint type.
- A 21-minute, four-person meeting transcribes in about a minute on a DGX Spark.

The job then carries two forms of the transcript:

- `transcript`: text with one paragraph per speaker turn, `Speaker 1: …` (plain text when only one
  speaker is heard). This is what the app stores and the summary reads; older app versions work
  with it unchanged.
- `transcript_json`: the structured document from
  [issue #12](https://github.com/metahertz/plaudlocal/issues/12): speakers, and segments with
  start/end times. Turn times are matched to the text by word position.

`POST /api/v1/jobs/text` also accepts an optional `transcript_json` (for speaker transcripts made
on the phone), stored with the job.

**Speaker names.** When a speaker transcript has two or more speakers, the summary model is asked
once per job, before the summary, to suggest a name for each speaker where the conversation
supports it, with a confidence and a short piece of evidence. Suggestions of at least 0.5 are
stored on the document's speakers as `suggestion`. They're never written into the transcript text,
and names the user has confirmed are never replaced. The summary prompt gets one line listing confirmed
names, and the suggestions marked unconfirmed. If the call fails, the job just continues without
names (the server log says why). Switch it off with **Suggest speaker names** under Summarization.

- **They're suggestions, not answers.** The transcriber can put two similar voices under one
  speaker label. The model is told to leave such a label unnamed, but the user has the final say.
- **Reasoning models:** the name call gets at most 5 minutes (less if **Timeout per call** is lower). With "thinking" on, a model can
  take longer than that on a long meeting, and the names are skipped. Thinking also makes
  summaries much slower. Turn it off in your model server if you can. In one test with
  Qwen3.8-27B, a 21-minute meeting's name call took 7 seconds with thinking off, and timed out
  with it on.

## HTTPS

The app refuses plain `http://` addresses, so recordings and the token never
travel unencrypted. The certificate must be one the iPhone trusts: a public
one (for example Let's Encrypt), or a private CA whose root you have installed
and trusted on the phone. Two ways to provide it:

- **Reverse proxy (recommended)**: keep the container on plain HTTP and put a
  TLS-terminating proxy in front. With Caddy, the whole configuration is:

  ```
  plaud.example.com {
      reverse_proxy plaud-remote:8080
  }
  ```

  Allow large request bodies and long uploads in the proxy; recordings can be
  tens of megabytes.
- **Built in**: mount a certificate and key into the container and set
  `PLAUD_REMOTE_TLS_CERT` and `PLAUD_REMOTE_TLS_KEY` to their paths. The server
  then serves HTTPS itself on its port.

## Summary model

A meeting summary is only as good as what the model can see at once. A decision
near the end often refers back to the start, and action items and concerns only
make sense from everything each person said. So the server gives the model the
**whole transcript in one request** whenever it fits the **Summary limit**
under Summarization.

**The default limit is 200,000 characters, sized for a model with a 64k-token
context window.** The context window itself is set on your model server, not
here (see below).

- A busy meeting produces about 55,000-65,000 characters of transcript an hour,
  and English is about 4 characters per token.
- 200,000 characters is about 50,000 tokens, roughly 3 hours of continuous
  talk. That leaves room in a 64k window for the instructions, the summary, and
  any "thinking" a reasoning model does before it answers.
- 64k is the smallest window that comfortably fits a 2-hour meeting
  (about 30-35k tokens of transcript plus that headroom), and it is within reach
  of a laptop (below).

For another context size, set the limit to about
`(context tokens - 10,000) x 4` characters. For example, a 32k model gets about
88,000, which is roughly 1.5 hours.

**The model server must actually use that context.** Many default to a smaller
window and silently drop the rest of a long prompt, and this server cannot ask
for a bigger one per request:

- Ollama: start it with `OLLAMA_CONTEXT_LENGTH=65536`. Its default depends on
  the version and the machine's memory, and on a laptop is often 4k or 32k
  tokens.
- LM Studio: set the model's context length to 65536 when loading it.
- vLLM: `--max-model-len 65536`. llama.cpp: `-c 65536`.

**Which model** is your choice; any OpenAI-compatible chat model works. As a
starting point we suggest **Qwen3.8-27B in a 4-bit build** for your runtime.
Only a quarter of its layers use full attention, which keeps a 64k context
cheap (an estimated 4 GB on top of roughly 16 GB of weights), so it fits on a
32 GB Mac. Try others and keep whichever summarizes your meetings best.
Bigger hosted models work too: raise the limit up to the maximum of 400,000.

A whole meeting in one request can take several minutes on a laptop, so raise
**Timeout per call** if summaries time out. The 4-characters-per-token figure
is for English; other languages can use more tokens per character, so leave
more headroom.

Transcripts over the limit are still summarized: part by part, each part with
as many summaries of the parts before it as fit within the limit, then merged.
The server logs a warning when that happens, because a split summary is
weaker.

Servers set up before this default was raised had 24,000 saved; the server
raises a saved 24,000 to 200,000 once, on upgrade, and logs that it did. A
value you set afterwards is kept. If your model's context is smaller than 64k
tokens, set the limit to match it.

## How a recording is processed

1. The app uploads the audio (`POST /api/v1/jobs`) and gets a job back.
2. A worker transcribes it, then summarizes the transcript in one pass. Only
   transcripts longer than the summary limit are summarized in parts (see
   [Summary model](#summary-model)).
3. The app polls the job (`GET /api/v1/jobs/<id>`) and stores the transcript
   and summary in the note. The brief, poster and export still run on the phone.

Behaviour worth knowing:

- **Retries**: failed model calls that look temporary (timeouts, HTTP 429/5xx)
  are retried with a growing delay, up to the configured number of attempts.
  A restart re-queues whatever was in progress without using up an attempt.
  A job that has been in progress at three restarts is failed instead, since it
  is probably what is stopping the server (for example, running out of memory on
  a very large recording). Retry in the web UI gives it a fresh start.
- **Summary failures don't lose the transcript**: the job finishes with the
  transcript and a reason, and the app falls back to on-device summarization.
- **No duplicate work**: uploading the same audio again returns the existing
  job while the model settings are unchanged. "Transcribe & summarize" in the
  app forces a fresh run.
- **Transcript-only jobs**: when the app will summarize on-device, or summaries
  are switched off in the app, it asks the server to skip the summary.
- **Audio is deleted** once a job finishes, unless you turn on "Keep the
  uploaded audio". Finished jobs are removed after 30 days by default.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PLAUD_REMOTE_ADMIN_PASSWORD` | generated | Web UI password for user `admin` |
| `PLAUD_REMOTE_PORT` | `8080` | Port to listen on |
| `PLAUD_REMOTE_DATA_DIR` | `/data` | Database and uploaded audio |
| `PLAUD_REMOTE_TLS_CERT`, `PLAUD_REMOTE_TLS_KEY` | unset | Certificate chain and key files, to serve HTTPS directly |
| `PLAUD_REMOTE_MAX_UPLOAD_MB` | `500` | Largest accepted recording |
| `PLAUD_REMOTE_WORKERS` | `1` | Recordings processed at the same time |
| `PLAUD_REMOTE_LOG_LEVEL` | `INFO` | Log verbosity |

Everything else is set in the web UI and stored in the data volume.

## Security

- The app API needs the bearer token; the web UI and admin API need the admin
  password. Regenerating the token in the web UI revokes the old one.
- The app connects over HTTPS only; see [HTTPS](#https). Use the web UI over
  HTTPS too, since it sends the admin password with every request.
- Model API keys and the app token are stored in the data volume, unencrypted,
  and model keys are never sent back to the browser. Protect the volume
  accordingly.
- Recordings and transcripts leave the phone when this feature is on. They go
  to this server and to whichever model endpoints you configure.

## App API

All requests need `Authorization: Bearer <token>`.

| Request | Purpose |
|---|---|
| `GET /api/v1/health` | Version, and whether transcription and summarization are configured |
| `POST /api/v1/jobs?client_ref=&title=&filename=&force=&summarize=` | Upload audio as the raw request body; `summarize=0` returns the transcript only |
| `POST /api/v1/jobs/text` | Summarize an existing transcript: `{"transcript": "...", "title": "...", "transcript_json": {...}}` (`transcript_json` optional) |
| `GET /api/v1/jobs/<id>` | Status, transcript, summary and errors; `transcript_json` for speaker transcripts |
| `DELETE /api/v1/jobs/<id>` | Remove a job and its data |

Job `status` is one of `queued`, `transcribing`, `summarizing`, `done`, `failed`.
`GET /healthz` needs no token and is used by the container health check.

## Tests

```bash
cd server
python -m unittest discover -s tests -v
```
