# Workflow Notes

This repository is worked on through a remote PC terminal workflow.

## Core Principles

- Research is the primary goal; coding should minimize overhead.
- Prefer small, targeted changes over broad refactors.
- Do not touch large dataset folders unless explicitly approved first.
- For important changes, notify before proceeding and leave a short change log after the work.

## Execution

- Run code on the remote PC terminal, not the local machine.
- Prefer direct interpreter paths over `conda activate`.
- Default Python path:

```powershell
D:\conda_envs\torch\python.exe
```

## Testing Policy

After code changes, run the most suitable lightweight verification for the project:

- `10-second run` for monitoring/apps/scripts where short live execution is the best smoke test.
- `single sample inference` for ML/inference projects where one realistic input is the best sanity check.

Choose the better fit case by case and report which one was used.

## Change Notification Rules

Notify before proceeding if the task would involve any of the following:

- large code structure changes
- creating new files
- deleting files
- modifying `json` or `yaml` config files
- touching large dataset folders

After the work, report:

- what changed
- what test was run
- any important risks or follow-up notes

## Deployment Constraints

- Support no-install, offline deployment when relevant.
- Keep lab PC constraints in mind for packaged tools and utilities.
