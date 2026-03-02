# GPU Worker — Runpod Serverless

Worker Docker genérico para execução de workloads GPU no Runpod Serverless.

O worker executa **operações atômicas** — toda lógica de pipeline (sequência de
operações, orquestração de batch, checkpoints) permanece no `ia-cadastro`.

## Arquitetura

```
┌─────────────────────────────────────────────────┐
│           Runpod Serverless Worker               │
│                                                  │
│  handler.py (runpod SDK)                         │
│  ├── operation: "health"    → GPU info + Ollama  │
│  ├── operation: "llm"       → Ollama localhost   │
│  ├── operation: "remove_bg" → rembg (GPU)        │
│  ├── operation: "upscale"   → Real-ESRGAN (GPU)  │
│  └── operation: "resize"    → Pillow             │
│                                                  │
│  Ollama (localhost:11434)                        │
│  ├── llama3.1:8b                                 │
│  ├── qwen2.5vl:3b                                │
│  └── qwen2.5vl:7b                                │
│                                                  │
│  GPU: NVIDIA RTX 4090 (24GB VRAM)                │
└─────────────────────────────────────────────────┘
```

## Operações

| Operação | Descrição | Input | Output |
|----------|-----------|-------|--------|
| `health` | GPU info, Ollama status, modelos | — | JSON status |
| `llm` | Inferência LLM (text-only ou multimodal) | prompt + config | texto gerado |
| `remove_bg` | Remoção de fundo (rembg + onnxruntime-gpu) | 1 imagem base64 | 1 imagem base64 |
| `upscale` | Upscale Real-ESRGAN x2/x4 (spandrel + PyTorch CUDA) | 1 imagem base64 | 1 imagem base64 |
| `resize` | Redimensionamento (Pillow) | 1 imagem base64 + target_size | 1 imagem base64 |

Todas as operações usam `/runsync` (síncrono). 1 imagem por request.

## Segurança

- HMAC-SHA256 por request (`X-Job-Signature` header validado no handler)
- API Key do Runpod (`Authorization: Bearer`)
- Spending limit no dashboard Runpod

## Estrutura

```
apps/gpu-worker/
├── handler.py              # Handler principal (runpod SDK)
├── operations/
│   ├── __init__.py
│   ├── health.py           # Health check
│   ├── llm.py              # LLM via Ollama localhost
│   ├── remove_bg.py        # Remoção de fundo (rembg GPU)
│   ├── upscale.py          # Upscale (Real-ESRGAN GPU)
│   └── resize.py           # Redimensionamento (Pillow)
├── security/
│   ├── __init__.py
│   └── hmac_validator.py   # Validação HMAC-SHA256
├── Dockerfile              # CUDA + Ollama + modelos baked (~20-25GB)
├── requirements.txt
├── start.sh                # Entrypoint (inicia Ollama + handler)
├── test_local.py           # Testes locais (runpod mock mode)
└── README.md
```

## Modelos baked na imagem Docker

Todos os modelos são incluídos na imagem Docker (sem Network Volume):

- **LLM**: llama3.1:8b (~4.7GB), qwen2.5vl:3b (~2GB), qwen2.5vl:7b (~4.5GB)
- **Upscale**: RealESRGAN_x2plus.pth (~64MB), RealESRGAN_x4plus.pth (~64MB)
- **Remoção de fundo**: BiRefNet ONNX (~170MB, baixado pelo rembg no build)

Tamanho total da imagem: ~20-25GB

## Teste local

```bash
cd apps/gpu-worker
pip install -r requirements.txt
python test_local.py
```

## Build e Deploy

```bash
# Build (precisa ~50GB disco, ~8GB RAM, NÃO precisa de GPU)
cd apps/gpu-worker
docker build -t seuuser/gpu-worker:v1.0.0 .

# Push pro Docker Hub
docker login
docker push seuuser/gpu-worker:v1.0.0

# No dashboard Runpod:
# 1. Serverless → New Endpoint
# 2. Imagem: seuuser/gpu-worker:v1.0.0
# 3. GPU: 4090 PRO (24GB)
# 4. Max workers: 3, Active workers: 0 (scale-to-zero)
# 5. Execution timeout: 600s
# 6. FlashBoot: habilitado
```

## Contrato da API

Ver documentação completa em `apps/gpu-orchestrator/docs/planejamento_gpu_runpod.md`, seção 8.
