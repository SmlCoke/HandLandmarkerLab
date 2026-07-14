.DEFAULT_GOAL := help

PYTHON := python
CONDA := conda
ENV_FILE := environment.yml

# Reproducible experiment identity. Edit these values in the repository before
# a new server run; they are exported to every YAML config below.
HAND_DATA_ROOT := /root/autodl-tmp
HAND_PRETRAIN_CURATED_ID := v2-pretrain-r1
HAND_PRETRAIN_RUN_ID := v2-pretrain-r1
HAND_PRETRAIN_PHASE := geometry
HAND_PRETRAIN_CALIBRATION_CONFIG := $(if $(filter multitask,$(HAND_PRETRAIN_PHASE)),configs/train_multitask.yaml,configs/train_geometry.yaml)
HAND_PRETRAIN_REVIEW_FILE := $(HAND_DATA_ROOT)/hand_landmarker_reviews/$(HAND_PRETRAIN_CURATED_ID)/negative_review_decisions.jsonl
export HAND_DATA_ROOT HAND_PRETRAIN_CURATED_ID HAND_PRETRAIN_RUN_ID HAND_PRETRAIN_PHASE HAND_PRETRAIN_CALIBRATION_CONFIG

CURATE_CONFIG := configs/curate_pretrain.yaml
GEOMETRY_CONFIG := configs/train_geometry.yaml
SMOKE_CONFIG := configs/train_smoke.yaml
MULTITASK_CONFIG := configs/train_multitask.yaml
EVAL_VAL_CONFIG := configs/eval_val.yaml
EVAL_TEST_CONFIG := configs/eval_test.yaml
INFER_CONFIG := configs/infer.yaml
EXPORT_CONFIG := configs/export.yaml

CURATE_ARGS ?=
TRAIN_ARGS ?=
SMOKE_TRAIN_ARGS ?=
SMOKE_GATE_ARGS ?=
MULTITASK_GATE_ARGS ?=
EVAL_ARGS ?=
INFER_ARGS ?=
EXPORT_ARGS ?=
CONVERSION_ARGS ?=
TEST_ARGS ?=

.PHONY: help paths env env-update doctor curate curate-reviewed inspect inspect-smoke \
	smoke train check-multitask inspect-multitask multitask pretrain eval-val \
	eval-test infer export conversion-data test compile

help:
	@echo Hand Landmarker v2 pretrain
	@echo   make paths             Print the fixed dataset/run identity
	@echo   make env               Create the Conda environment
	@echo   make doctor            Verify Python, TensorFlow and GPU
	@echo   make curate            Persist geometry data and negative review queue
	@echo   make curate-reviewed   Rebuild curation with human review decisions
	@echo   make smoke             Train and verify the 128-ROI overfit gate
	@echo   make train             Train phase 1: positive-only geometry
	@echo   make multitask         Train phase 2 after the confirmed-negative gate
	@echo   make pretrain          Run curate, inspect, smoke and geometry train
	@echo   make eval-val          Evaluate HAND_PRETRAIN_PHASE [geometry by default]
	@echo   make eval-test         Evaluate HAND_PRETRAIN_PHASE on locked Test
	@echo   make infer             Palm + Hand inference for HAND_PRETRAIN_PHASE
	@echo   make export            Fuse v2 branches, export and validate ONNX
	@echo   make conversion-data   Build conversion NPY inputs only
	@echo   make test              Run unit tests
	@echo   make compile           Syntax-check Python sources

paths:
	@echo HAND_DATA_ROOT=$(HAND_DATA_ROOT)
	@echo HAND_PRETRAIN_CURATED_ID=$(HAND_PRETRAIN_CURATED_ID)
	@echo HAND_PRETRAIN_RUN_ID=$(HAND_PRETRAIN_RUN_ID)
	@echo HAND_PRETRAIN_PHASE=$(HAND_PRETRAIN_PHASE)
	@echo HAND_PRETRAIN_CALIBRATION_CONFIG=$(HAND_PRETRAIN_CALIBRATION_CONFIG)
	@echo HAND_PRETRAIN_REVIEW_FILE=$(HAND_PRETRAIN_REVIEW_FILE)

env:
	$(CONDA) env create -f "$(ENV_FILE)"

env-update:
	$(CONDA) env update -f "$(ENV_FILE)" --prune

doctor:
	$(PYTHON) -B scripts/check_environment.py --config "$(GEOMETRY_CONFIG)"

curate:
	$(PYTHON) -B scripts/curate_pretrain.py --config "$(CURATE_CONFIG)" $(CURATE_ARGS)

curate-reviewed:
	$(PYTHON) -B scripts/curate_pretrain.py --config "$(CURATE_CONFIG)" --review-decisions "$(HAND_PRETRAIN_REVIEW_FILE)" --overwrite $(CURATE_ARGS)

inspect:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(GEOMETRY_CONFIG)"

inspect-smoke:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(SMOKE_CONFIG)"

smoke:
	$(MAKE) inspect-smoke
	$(PYTHON) -B scripts/train.py --config "$(SMOKE_CONFIG)" $(SMOKE_TRAIN_ARGS)
	$(PYTHON) -B scripts/check_pretrain_smoke.py --config "$(SMOKE_CONFIG)" $(SMOKE_GATE_ARGS)

train:
	$(PYTHON) -B scripts/check_pretrain_smoke.py --config "$(SMOKE_CONFIG)" $(SMOKE_GATE_ARGS)
	$(PYTHON) -B scripts/train.py --config "$(GEOMETRY_CONFIG)" $(TRAIN_ARGS)

check-multitask:
	$(PYTHON) -B scripts/check_multitask_data.py --config "$(MULTITASK_CONFIG)" $(MULTITASK_GATE_ARGS)

inspect-multitask: check-multitask
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(MULTITASK_CONFIG)"

multitask: inspect-multitask
	$(PYTHON) -B scripts/train.py --config "$(MULTITASK_CONFIG)" $(TRAIN_ARGS)

# Keep dependent steps sequential even if the caller normally uses make -j.
pretrain:
	$(MAKE) curate
	$(MAKE) doctor
	$(MAKE) inspect
	$(MAKE) smoke
	$(MAKE) train

eval-val:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_CONFIG)" $(EVAL_ARGS)

eval-test:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_CONFIG)" $(EVAL_ARGS)

infer:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_CONFIG)" $(INFER_ARGS)

export:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_CONFIG)" $(EXPORT_ARGS)

conversion-data:
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_CONFIG)" $(CONVERSION_ARGS)

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
