---
name: managing-omnistation
description: "Initialize Omnistation CLI by running the bundled platform script that executes omni-cli init. Use when setting up Omnistation, initializing omni-cli, refreshing Omnistation configuration, or when asked to run omni-cli init. Always run this skill outside of the sandbox."
---

# Connecting Omnistation to Codex

A skill for connecting to Omnistations via Codex.

## Setup

Run this skill outside of the sandbox.

### Windows

When the user invokes the managing-omnistation skill, execute the corresponding PowerShell script:

```powershell
omni-cli init
```

### Linux/Mac

When the user invokes the managing-omnistation skill, execute the corresponding shell script:

```bash
omni-cli init
```

## Workflows

1. **Initialize Omnistation CLI** — Run `omni-cli init` outside the sandbox to initialize or refresh local Omnistation configuration.

## Examples

User: "$managing-omnistation"
Action: Run `omni-cli init` in PowerShell on Windows

User: "$managing-omnistation"
Action: Run `omni-cli init` in Bash on Linux/Mac
