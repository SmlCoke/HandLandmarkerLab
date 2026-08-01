# AGENTS.md

## I. Purpose

This file defines the repository-wide instructions for coding agents.

Agents must follow these instructions when inspecting, modifying, testing, or
documenting this repository.

The primary goals are:

1. Preserve repository correctness and maintainability.
2. Make only changes required by the current task.
3. Avoid destructive, unrelated, or speculative modifications.

---

## II. Instruction Priority

Follow instructions in this order:

1. Explicit instructions in the current user request.
2. The nearest applicable `AGENTS.override.md`.
3. The nearest applicable `AGENTS.md`.
4. This repository-level `AGENTS.md`.
5. Existing project conventions inferred from nearby code.

Higher-priority instructions override lower-priority instructions.

Do not interpret a previous task's temporary authorization as authorization for
the current task.

When two applicable instructions conflict and the conflict cannot be resolved
safely, stop before modifying files and explain the conflict.

---

## III. Repository Overview

### 3.1 Project purpose

We are developing a real-time sign language recognition system designed for low-compute (0.8 TOPS) edge devices. The system comprises a pipeline of three models connected in series:
1. Palm Detector: Processes images captured by the camera and outputs palm bounding box coordinates along with the coordinates of two auxiliary points.
2. Hand Landmarker: Performs inference on the Hand ROI defined by the palm bounding box to obtain the coordinates of 21 skeletal keypoints, as well as confidence scores for hand presence and handedness (left/right).
3. Gloss Translator: An isolated sign classification model that utilizes the outputs from the Palm Detector and Hand Landmarker; it maintains a temporal window of a specific length, performs temporal modeling, and outputs the isolated sign corresponding to the movements within that timeframe.

This repository contains the training system for the Hand Landmarker model.

### 3.2 Entry-point documents

- `docs\training_system\HLML_training_workflow.md`: Referred to as the "workflow" document; it explains the current system's workflows and procedures rather than serving merely as an operation manual.
- `docs\training_system\HLML_quick_start.md`: Referred to as the "quick_start" document; a simplified version of the "workflow" document containing instructions for executing the full process—ideal for getting started quickly.
- `docs\training_system\HLML_current_training_status.md`: Referred to as the "current_status" document; it records the model's current training status. The content currently reflects the state as of July 19, 2026—specifically, the version preceding the final regional competition—offering some reference value.
- `docs\training_system\HLML_next_step_plan.md`: Referred to as the "next_step_plan" document; it outlines the training plans and objectives for the next phase. The current content details the training plan and goals for the present stage (the National Finals).

## IV. General Working Rules

### 4.1 Docs Modifying Rules

- The "workflow" document strictly records the commands, content, and underlying principles for each operational step of the system; it is independent of the system's historical state, the server-side model training status, and the project's future plans.It is necessary to explain the command and input (including directory locations) for each step, the actions performed, the output (including directory locations), and the rationale behind parameter adjustments in the YAML configuration file. Please keep this principle in mind when making modifications.
- The "quick_start" document is a simplified version of the "workflow" document; it contains only the commands for each operational step and omits explanations of underlying principles. Include the name of the process stage for each step and briefly describe the inputs and outputs. Please keep this principle in mind when making modifications. 
- The "current_status" document records the current state of the system and the performance of the server-side model training. Please keep this principle in mind when making modifications.
- The "next_step_plan" document outlines the plan for the next phase. Please keep this principle in mind when making modifications.

These four documents have distinct roles and independent content; each should avoid extensively detailing the information covered in the others.
These four documents serve as the primary interface documentation for the repository and are critical; they must be kept synchronized whenever there are subsequent updates to code, configurations, or other documentation.

### 4.2 Principle of simplification

1. Whether during manual operations or automated execution of the repository's programs, performing a hash check (SHA256) at every step is prohibited, as this results in significant waste of time and excessive disk space usage. Datasets from different sources can be effectively isolated based on details such as their source names.
2. During each operation, the agent should perform only the tasks explicitly required by the prompt; maintaining simplicity avoids unnecessary, redundant auditing and verification, which would otherwise waste time.