.DEFAULT_GOAL := help

PYTHON := python
CONDA := conda
ENV_FILE := environment.yml

# The only experiment inputs that operators edit between runs.
# HLML-2.0 is the deployed data-layout namespace. It is intentionally stable
# across repository tags and isolates TrainFab from the separate DatasetFab.
HAND_TRAIN_ROOT := /root/autodl-tmp/TrainFab/HLML-2.0
HAND_PRETRAIN_ID := v2-pretrain-r3
export HAND_TRAIN_ROOT HAND_PRETRAIN_ID

CURATE_CONFIG := configs/curate_pretrain.yaml
GEOMETRY_CONFIG := configs/train_geometry.yaml
SMOKE_CONFIG := configs/train_smoke.yaml
MULTITASK_CONFIG := configs/train_multitask.yaml
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
	eval-val-geometry eval-test-geometry eval-val-multitask eval-test-multitask \
	infer-geometry infer-multitask export-geometry export-multitask \
	conversion-data-geometry conversion-data-multitask \
	test test-unit test-export-preflight compile

help:
	@echo Hand Landmarker v2 pretrain - TrainFab layout HLML-2.0
	@echo   make paths                       Print the fixed training root and experiment ID
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
	@echo   make test                        Run unit tests, then build the untrained ONNX conversion bundle
	@echo   make test-unit                   Run unit tests only
	@echo   make test-export-preflight       Build the untrained ONNX conversion bundle only
	@echo   make compile                     Syntax-check Python sources

paths:
	@echo HAND_TRAIN_ROOT=$(HAND_TRAIN_ROOT)
	@echo HAND_PRETRAIN_ID=$(HAND_PRETRAIN_ID)
	@echo CURATED_ROOT=$(HAND_TRAIN_ROOT)/train_pretrain_curated/$(HAND_PRETRAIN_ID)
	@echo RUN_ROOT=$(HAND_TRAIN_ROOT)/hand_landmarker_runs/$(HAND_PRETRAIN_ID)

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

eval-val-geometry eval-test-geometry infer-geometry export-geometry conversion-data-geometry: export HAND_PRETRAIN_PHASE := geometry
export-geometry conversion-data-geometry: export HAND_PRETRAIN_CALIBRATION_CONFIG := configs/train_geometry.yaml

eval-val-multitask eval-test-multitask infer-multitask export-multitask conversion-data-multitask: export HAND_PRETRAIN_PHASE := multitask
export-multitask conversion-data-multitask: export HAND_PRETRAIN_CALIBRATION_CONFIG := configs/train_multitask.yaml

eval-val-geometry eval-val-multitask:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_CONFIG)" $(EVAL_ARGS)

eval-test-geometry eval-test-multitask:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_CONFIG)" $(EVAL_ARGS)

infer-geometry infer-multitask:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_CONFIG)" $(INFER_ARGS)

export-geometry export-multitask:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_CONFIG)" $(EXPORT_ARGS)

conversion-data-geometry conversion-data-multitask:
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_CONFIG)" $(CONVERSION_ARGS)

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)
	$(PYTHON) -B scripts/build_export_preflight.py --config "$(PREFLIGHT_EXPORT_CONFIG)"

test-unit:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

test-export-preflight:
	$(PYTHON) -B scripts/build_export_preflight.py --config "$(PREFLIGHT_EXPORT_CONFIG)"

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
