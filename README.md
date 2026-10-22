# AirLLM-Go

AirLLM-Go is a polished, local-first AI workspace with a Claude-inspired interface and an [AirLLM](https://github.com/lyogavin/airllm) inference backend. It keeps conversations on your machine and makes model downloads and runtime progress visible.

## Features

- Local chat powered by AirLLM and Hugging Face models
- Searchable model picker with per-conversation model memory
- First-run onboarding for cache drive, default model, and disk-space awareness
- Persistent conversation history in a local JSON file
- Live terminal and model-loading progress panel
- Text/code file attachments
- Optional local image generation through Diffusers
- Premium themes, liquid-glass surfaces, animated transitions, keyboard shortcuts, and reduced-motion support
- Responsive layout with a Sierpinski triangle brand mark

## Open on Windows

Double-click `run_air.bat`. It starts the local server and opens the app in your browser.

Or launch from PowerShell:

```powershell
.\.airenv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) if the browser does not open automatically. AirLLM-Go uses Python 3.14.7 in the supplied environment and listens on `127.0.0.1:7860` by default.

## First-run setup

1. Start the app.
2. Choose a model-cache drive. If C: is full, choose D: or another drive with several GB available.
3. Choose a default model. Start with `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-4B` to verify the setup.
4. Model files are normally stored in `D:\AirLLM-Go\huggingface` after setup.

The selection is saved in `airllm-go-config.json`, which is machine-specific and intentionally ignored by Git.

## Model loading

The first request downloads files and prepares AirLLM layer shards. This can take minutes on CPU. Use the Terminal tab to watch live output. Later launches reuse the cache.

| Model | Use case | Footprint |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B` | Setup verification and quick tests | Low |
| `Qwen/Qwen3-4B` | Everyday local chat | Medium |
| `Qwen/Qwen3-8B` | Higher quality local chat | High |
| Dolphin and other community checkpoints | Alternative / unfiltered models | High |

Actual requirements depend on model size, compression, RAM, and GPU availability.

## Optional image generation

```powershell
.\.airenv\Scripts\python.exe -m pip install -r requirements-image.txt
```

Then use the ✦ composer button. Image models are large and CPU rendering can be slow.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Flask app, UI, AirLLM integration, uploads, and API routes |
| `run_air.bat` | One-click Windows launcher |
| `requirements.txt` | Core dependencies |
| `requirements-image.txt` | Optional Diffusers dependencies |
| `conversations.json` | Persistent local history; ignored by Git |
| `airllm-go-config.json` | Cache/default-model setup; ignored by Git |
| `airllm-go-terminal.log` | Runtime log; ignored by Git |

## Troubleshooting

### Page does not open

Confirm the launcher is running and visit `http://127.0.0.1:7860`. Use Ctrl+F5 if an old interface is cached.

### Not enough space

Choose a different cache drive in onboarding. Avoid putting large model caches on a nearly-full C: drive.

### The UI appears stuck at `0/35`

AirLLM is usually reading or converting the first model shard. This is especially slow on CPU. The Terminal tab remains available while the model loads; use a smaller model for the first smoke test.

### Terminal is blank

Use Ctrl+F5. The backend exposes live output at `/api/terminal` and must be running on port 7860.

## Publish to GitHub

The project is prepared for GitHub, but the connected GitHub integration cannot create a repository. Create an empty public repository named `AirLLM-Go` under `PrattaySarkar` at <https://github.com/new>. Do not add a README, `.gitignore`, or license there.

Then run these commands from this folder:

```powershell
git init
git branch -M main
git add app.py README.md requirements.txt requirements-image.txt run_air.bat .gitignore
git commit -m "Initial AirLLM-Go release"
git remote add origin https://github.com/PrattaySarkar/AirLLM-Go.git
git push -u origin main
```

Never commit `.airenv`, downloaded model files, local conversations, cache configuration, uploads, or runtime logs. The included `.gitignore` protects them.

## License

Add a license before public distribution. AirLLM-Go depends on third-party projects with their own licenses; review their terms before redistribution.
