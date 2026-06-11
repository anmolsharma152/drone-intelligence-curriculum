# Phase 04: Systems Engineering & ML Pipeline

## Overview

Single-drone sim is a prototype. Multi-drone, networked, with a data
pipeline and model serving — that's a system.

This phase has three tracks:

```
Infrastructure:          ML Pipeline (🤖):
  Multi-drone ECS  ──►  Feature engineering from telemetry
  UDP protocol      ──►  Model serving (online inference)
  Data logging      ──►  Online learning loop
       │                        │
       └────────┬───────────────┘
                ▼
         Production System
         (multiple drones, 1 training server,
          live inference, streaming telemetry)
```

## Files

| # | File | Description | AI? |
|---|---|---|---|
| 01 | `01-multi-drone-architecture.md` | Multi-drone system design, comms | |
| 02 | `02-ecs-pattern.md` | Entity-Component-System architecture | |
| 03 | `03-udp-command-protocol.md` | UDP command/telemetry protocol | |
| 04 | `04-data-logging-telemetry.md` | Structured logging, replay | |
| 05 | `05-feature-engineering-pipeline.md` | 🤖 Feature extraction from raw telemetry | ✓ |
| 06 | `06-model-serving-infrastructure.md` | 🤖 Deploy trained policy for live inference | ✓ |
| 07 | `07-online-learning-loop.md` | 🤖 Continuous adaptation mid-flight | ✓ |
| — | `GATE-03.md` | Verification gate | |

## Learning Objectives

1. Design multi-drone system with Entity-Component-System pattern
2. Implement UDP-based command protocol for remote drone control
3. Build structured telemetry logging with replay capability
4. **Extract features from raw sensor data for ML consumption** 🤖
5. **Serve trained policies in a low-latency inference server** 🤖
6. **Implement online learning that adapts mid-flight** 🤖

## Prerequisites

- Phase 03 completed (gate passed)
- PyTorch + numpy
- Basic understanding of UDP networking (Python `socket` module)

## Embedded Notes

- ECS maps cleanly to embedded C++ on a Pixhawk/STM32
- UDP protocol could run over ESP-NOW or LoRa on a Pi Pico
- Model inference on edge: TensorFlow Lite, ONNX Runtime, or custom
  MCU inference engine

**Next:** `01-multi-drone-architecture.md`
