.DEFAULT_GOAL := help

PYTHON := python
CONDA := conda
ENV_FILE := environment.yml

# The only experiment inputs that operators edit between runs.
# TrainFab stores only generated metadata/runs. DatasetFab is the read-only,
# reproducible image warehouse and is referenced directly wherever possible.
HAND_TRAIN_ROOT ?= /root/autodl-tmp/TrainFab/HLML-3.0
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
HAND_PRETRAIN_ID ?= v3-pretrain-r1
HAND_FINETUNE_ID ?= v3-finetune-r1
FINETUNE_EXPERIMENT_ID ?= $(HAND_FINETUNE_ID)
FINETUNE_PROFILE ?= data_only
FINETUNE_ROUND_ID ?= r01
FINETUNE_GOLD_BUDGET ?=
NEW_RECORDED_SOURCE_ID ?=
BASELINE_FINETUNE_ID ?= v3-finetune-r1
CANDIDATE_FINETUNE_ID ?= $(FINETUNE_EXPERIMENT_ID)
ANALYSIS_OVERWRITE ?= 0
export HAND_TRAIN_ROOT HAND_DATASET_ROOT HAND_PRETRAIN_ID HAND_FINETUNE_ID
export FINETUNE_EXPERIMENT_ID FINETUNE_PROFILE

CURATE_CONFIG := configs/curate_pretrain.yaml
GEOMETRY_CONFIG := configs/train_geometry.yaml
SMOKE_CONFIG := configs/train_smoke.yaml
MULTITASK_CONFIG := configs/train_multitask.yaml
PREPARE_FINETUNE_CONFIG := configs/prepare_finetune_sources.yaml
FINETUNE_CURATE_CONFIG := configs/curate_finetune.yaml
FINETUNE_CONFIG := configs/train_finetune.yaml
FINETUNE_SMOKE_CONFIG := configs/train_finetune_smoke.yaml
EVAL_VAL_CONFIG := configs/eval_val.yaml
EVAL_TEST_CONFIG := configs/eval_test.yaml
INFER_CONFIG := configs/infer.yaml
EXPORT_CONFIG := configs/export.yaml
PREFLIGHT_EXPORT_CONFIG := configs/export_preflight.yaml

CURATE_ARGS ?=
GEOMETRY_ARGS ?=
MULTITASK_ARGS ?=
SMOKE_TRAIN_ARGS ?=
SMOKE_GATE_ARGS ?=
MULTITASK_GATE_ARGS ?=
PREPARE_FINETUNE_ARGS ?=
CURATE_FINETUNE_ARGS ?=
FINETUNE_SOURCE_GATE_ARGS ?=
FINETUNE_DATA_GATE_ARGS ?=
FINETUNE_TRAIN_ARGS ?=
FINETUNE_SMOKE_TRAIN_ARGS ?=
FINETUNE_SMOKE_GATE_ARGS ?=
EVAL_ARGS ?=
INFER_ARGS ?=
EXPORT_ARGS ?=
CONVERSION_ARGS ?=
TEST_ARGS ?=

.PHONY: help paths env-create env-update doctor \
	pretrain-curate pretrain-curate-reviewed \
	inspect-geometry inspect-geometry-smoke inspect-multitask \
	pretrain-geometry-smoke check-geometry-smoke pretrain-geometry \
	check-multitask-data pretrain-multitask \
	prepare-finetune-sources prepare-finetune-round check-finetune-sources finetune-curate \
	check-finetune-data finetune-smoke check-finetune-smoke \
	inspect-finetune finetune-train \
	eval-val-geometry eval-test-geometry eval-val-multitask eval-test-multitask \
	eval-val-finetune eval-test-finetune infer-finetune export-finetune conversion-data-finetune \
	infer-geometry infer-multitask export-geometry export-multitask \
	conversion-data-geometry conversion-data-multitask \
	analyze-finetune-errors compare-finetune-runs test test-unit test-export-preflight compile

help:
	@echo Hand Landmarker Lab 3.0 - DatasetFab read-only + TrainFab generated workspace
	@echo   make paths                       Print the fixed training root and both experiment IDs
	@echo   make env-create                  Create the documented Conda environment
	@echo   make env-update                  Reconcile the documented Conda environment
	@echo   make doctor                      Verify Python, TensorFlow and GPU
	@echo   make pretrain-curate             Persist geometry data and create the visual review folder
	@echo   make pretrain-curate-reviewed    Confirm retained review images and rebuild multitask data
	@echo   make inspect-geometry            Audit geometry Train, Val and locked Test
	@echo   make inspect-geometry-smoke      Audit the fixed 128-ROI smoke set
	@echo   make pretrain-geometry-smoke     Train and verify the geometry smoke gate
	@echo   make check-geometry-smoke        Recheck an existing geometry smoke run
	@echo   make pretrain-geometry           Train geometry after the smoke gate
	@echo   make check-multitask-data        Verify human-confirmed negative evidence
	@echo   make inspect-multitask           Audit multitask Train, Val and locked Test
	@echo   make pretrain-multitask          Train multitask from geometry best
	@echo   make prepare-finetune-sources    Select hard/disagreement Gold requests and replay
	@echo   make prepare-finetune-round      Freeze one cumulative-disjoint Gold selection round
	@echo   make analyze-finetune-errors     Compare Val/infer failures and render at most 40 overlays
	@echo   make compare-finetune-runs       Produce a paired candidate-versus-baseline report
	@echo   make check-finetune-sources      Authenticate HLMF Gold sources and aggregate
	@echo   make finetune-curate             Merge authenticated Gold with replay
	@echo   make check-finetune-data         Verify the immutable finetune snapshot and mix
	@echo   make finetune-smoke              Train and verify the 256-ROI finetune smoke gate
	@echo   make check-finetune-smoke        Recheck an existing finetune smoke run
	@echo   make inspect-finetune             Audit finetune Train, Val and locked Test
	@echo   make finetune-train               Train finetune after the smoke gate
	@echo   make eval-val-geometry           Evaluate geometry on Val
	@echo   make eval-test-geometry          Evaluate geometry on locked Test
	@echo   make eval-val-multitask          Evaluate multitask on Val
	@echo   make eval-test-multitask         Evaluate multitask on locked Test
	@echo   make infer-geometry              Palm + geometry Hand inference
	@echo   make infer-multitask             Palm + multitask Hand inference
	@echo   make export-geometry             Fuse and export geometry ONNX
	@echo   make export-multitask            Fuse and export multitask ONNX
	@echo   make conversion-data-geometry    Build geometry conversion NPY inputs
	@echo   make conversion-data-multitask   Build multitask conversion NPY inputs
	@echo   make eval-val-finetune            Evaluate finetune on Val
	@echo   make eval-test-finetune           Evaluate finetune on locked Test
	@echo   make infer-finetune               Palm + finetune Hand inference
	@echo   make export-finetune              Fuse and export finetune ONNX
	@echo   make conversion-data-finetune     Build finetune conversion NPY inputs
	@echo   make test                        Run unit tests, then build the untrained ONNX conversion bundle
	@echo   make test-unit                   Run unit tests only
	@echo   make test-export-preflight       Build the untrained ONNX conversion bundle only
	@echo   make compile                     Syntax-check Python sources

paths:
	@echo HAND_TRAIN_ROOT=$(HAND_TRAIN_ROOT)
	@echo HAND_DATASET_ROOT=$(HAND_DATASET_ROOT)
	@echo HAND_PRETRAIN_ID=$(HAND_PRETRAIN_ID)
	@echo HAND_FINETUNE_ID=$(HAND_FINETUNE_ID)
	@echo FINETUNE_EXPERIMENT_ID=$(FINETUNE_EXPERIMENT_ID)
	@echo FINETUNE_PROFILE=$(FINETUNE_PROFILE)
	@echo CURATED_ROOT=$(HAND_TRAIN_ROOT)/train_pretrain_curated/$(HAND_PRETRAIN_ID)
	@echo RUN_ROOT=$(HAND_TRAIN_ROOT)/hand_landmarker_runs/$(HAND_PRETRAIN_ID)
	@echo FINETUNE_WORKSPACE=$(HAND_TRAIN_ROOT)/finetune/$(HAND_FINETUNE_ID)
	@echo FINETUNE_RUN_ROOT=$(HAND_TRAIN_ROOT)/hand_landmarker_runs/$(HAND_FINETUNE_ID)

env-create:
	$(CONDA) env create -f "$(ENV_FILE)"

env-update:
	$(CONDA) env update -f "$(ENV_FILE)" --prune

doctor:
	$(PYTHON) -B scripts/check_environment.py --config "$(GEOMETRY_CONFIG)"

pretrain-curate:
	$(PYTHON) -B scripts/curate_pretrain.py --config "$(CURATE_CONFIG)" $(CURATE_ARGS)

pretrain-curate-reviewed:
	$(PYTHON) -B scripts/curate_pretrain.py --config "$(CURATE_CONFIG)" --finalize-retained-review --overwrite $(CURATE_ARGS)

inspect-geometry:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(GEOMETRY_CONFIG)"

inspect-geometry-smoke:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(SMOKE_CONFIG)"

pretrain-geometry-smoke: inspect-geometry-smoke
	$(PYTHON) -B scripts/train.py --config "$(SMOKE_CONFIG)" $(SMOKE_TRAIN_ARGS)
	$(PYTHON) -B scripts/check_pretrain_smoke.py --config "$(SMOKE_CONFIG)" $(SMOKE_GATE_ARGS)

check-geometry-smoke:
	$(PYTHON) -B scripts/check_pretrain_smoke.py --config "$(SMOKE_CONFIG)" $(SMOKE_GATE_ARGS)

pretrain-geometry: check-geometry-smoke
	$(PYTHON) -B scripts/train.py --config "$(GEOMETRY_CONFIG)" $(GEOMETRY_ARGS)

check-multitask-data:
	$(PYTHON) -B scripts/check_multitask_data.py --config "$(MULTITASK_CONFIG)" $(MULTITASK_GATE_ARGS)

inspect-multitask:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(MULTITASK_CONFIG)"

pretrain-multitask: check-multitask-data inspect-multitask
	$(PYTHON) -B scripts/train.py --config "$(MULTITASK_CONFIG)" $(MULTITASK_ARGS)

prepare-finetune-sources:
	$(PYTHON) -B scripts/prepare_finetune_sources.py --config "$(PREPARE_FINETUNE_CONFIG)" $(PREPARE_FINETUNE_ARGS)

prepare-finetune-round:
	$(if $(strip $(FINETUNE_GOLD_BUDGET)),,$(error FINETUNE_GOLD_BUDGET is required))
	$(PYTHON) -B scripts/prepare_finetune_round.py --config "$(PREPARE_FINETUNE_CONFIG)" --round-id "$(FINETUNE_ROUND_ID)" --gold-budget "$(FINETUNE_GOLD_BUDGET)" $(if $(strip $(NEW_RECORDED_SOURCE_ID)),--new-recorded-source-id "$(NEW_RECORDED_SOURCE_ID)",)

check-finetune-sources:
	$(PYTHON) -B scripts/check_finetune_sources.py --config "$(FINETUNE_CURATE_CONFIG)" $(FINETUNE_SOURCE_GATE_ARGS)

finetune-curate: check-finetune-sources
	$(PYTHON) -B scripts/curate_finetune.py --config "$(FINETUNE_CURATE_CONFIG)" $(CURATE_FINETUNE_ARGS)

check-finetune-data:
	$(PYTHON) -B scripts/check_finetune_data.py --config "$(FINETUNE_CONFIG)" $(FINETUNE_DATA_GATE_ARGS)

finetune-smoke: check-finetune-data inspect-finetune
	$(PYTHON) -B scripts/train.py --config "$(FINETUNE_SMOKE_CONFIG)" $(FINETUNE_SMOKE_TRAIN_ARGS)
	$(PYTHON) -B scripts/check_finetune_smoke.py --smoke-config "$(FINETUNE_SMOKE_CONFIG)" --full-config "$(FINETUNE_CONFIG)" $(FINETUNE_SMOKE_GATE_ARGS)

check-finetune-smoke:
	$(PYTHON) -B scripts/check_finetune_smoke.py --smoke-config "$(FINETUNE_SMOKE_CONFIG)" --full-config "$(FINETUNE_CONFIG)" $(FINETUNE_SMOKE_GATE_ARGS)

inspect-finetune: check-finetune-data
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(FINETUNE_CONFIG)"

finetune-train: check-finetune-data inspect-finetune check-finetune-smoke
	$(PYTHON) -B scripts/train.py --config "$(FINETUNE_CONFIG)" $(FINETUNE_TRAIN_ARGS)

eval-val-geometry eval-test-geometry infer-geometry export-geometry conversion-data-geometry: export HAND_EXPERIMENT_ID := $(HAND_PRETRAIN_ID)
eval-val-geometry eval-test-geometry infer-geometry export-geometry conversion-data-geometry: export HAND_RUN_PHASE := geometry
eval-val-geometry eval-test-geometry infer-geometry export-geometry conversion-data-geometry: export HAND_MODEL_STAGE := pretrain
export-geometry conversion-data-geometry: export HAND_TRAIN_CONFIG := configs/train_geometry.yaml

eval-val-multitask eval-test-multitask infer-multitask export-multitask conversion-data-multitask: export HAND_EXPERIMENT_ID := $(HAND_PRETRAIN_ID)
eval-val-multitask eval-test-multitask infer-multitask export-multitask conversion-data-multitask: export HAND_RUN_PHASE := multitask
eval-val-multitask eval-test-multitask infer-multitask export-multitask conversion-data-multitask: export HAND_MODEL_STAGE := pretrain
export-multitask conversion-data-multitask: export HAND_TRAIN_CONFIG := configs/train_multitask.yaml

eval-val-finetune eval-test-finetune infer-finetune export-finetune conversion-data-finetune: export HAND_EXPERIMENT_ID := $(FINETUNE_EXPERIMENT_ID)
eval-val-finetune eval-test-finetune infer-finetune export-finetune conversion-data-finetune: export HAND_RUN_PHASE := finetune
eval-val-finetune eval-test-finetune infer-finetune export-finetune conversion-data-finetune: export HAND_MODEL_STAGE := finetune
export-finetune conversion-data-finetune: export HAND_TRAIN_CONFIG := configs/train_finetune.yaml

eval-val-geometry eval-val-multitask eval-val-finetune:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_CONFIG)" $(EVAL_ARGS)

eval-test-geometry eval-test-multitask eval-test-finetune:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_CONFIG)" $(EVAL_ARGS)

infer-geometry infer-multitask infer-finetune:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_CONFIG)" $(INFER_ARGS)

export-geometry export-multitask export-finetune:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_CONFIG)" $(EXPORT_ARGS)

conversion-data-geometry conversion-data-multitask conversion-data-finetune:
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_CONFIG)" $(CONVERSION_ARGS)

analyze-finetune-errors:
	$(PYTHON) -B scripts/analyze_finetune.py --work-root "$(HAND_TRAIN_ROOT)" --baseline-id "$(BASELINE_FINETUNE_ID)" --candidate-id "$(CANDIDATE_FINETUNE_ID)" --labels "$(HAND_TRAIN_ROOT)/val_merged/05_labels/hand_validation_labels.jsonl" --output-dir "$(HAND_TRAIN_ROOT)/hand_landmarker_runs/$(CANDIDATE_FINETUNE_ID)/analysis/error_audit" --overlay-limit 40 $(if $(filter 1 true yes on,$(ANALYSIS_OVERWRITE)),--overwrite,)

compare-finetune-runs:
	$(PYTHON) -B scripts/analyze_finetune.py --work-root "$(HAND_TRAIN_ROOT)" --baseline-id "$(BASELINE_FINETUNE_ID)" --candidate-id "$(CANDIDATE_FINETUNE_ID)" --labels "$(HAND_TRAIN_ROOT)/val_merged/05_labels/hand_validation_labels.jsonl" --output-dir "$(HAND_TRAIN_ROOT)/hand_landmarker_runs/$(CANDIDATE_FINETUNE_ID)/analysis/compare_vs_$(BASELINE_FINETUNE_ID)" --overlay-limit 40 $(if $(filter 1 true yes on,$(ANALYSIS_OVERWRITE)),--overwrite,)

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)
	$(PYTHON) -B scripts/build_export_preflight.py --config "$(PREFLIGHT_EXPORT_CONFIG)"

test-unit:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

test-export-preflight:
	$(PYTHON) -B scripts/build_export_preflight.py --config "$(PREFLIGHT_EXPORT_CONFIG)"

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
