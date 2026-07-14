.DEFAULT_GOAL := help

PYTHON ?= python
CONDA ?= conda
ENV_FILE ?= environment.yml
MODEL_STAGE ?= pretrain

CURATE_PRETRAIN_CONFIG ?= configs/curate_pretrain.yaml
TRAIN_PRETRAIN_CONFIG ?= configs/train_pretrain.yaml
TRAIN_PRETRAIN_SMOKE_CONFIG ?= configs/train_pretrain_smoke.yaml
TRAIN_FINETUNE_CONFIG ?= configs/train_finetune.yaml
EVAL_VAL_PRETRAIN_CONFIG ?= configs/eval_val_pretrain.yaml
EVAL_VAL_FINETUNE_CONFIG ?= configs/eval_val_finetune.yaml
EVAL_TEST_PRETRAIN_CONFIG ?= configs/eval_test_pretrain.yaml
EVAL_TEST_FINETUNE_CONFIG ?= configs/eval_test_finetune.yaml
INFER_PRETRAIN_CONFIG ?= configs/infer_pretrain.yaml
INFER_FINETUNE_CONFIG ?= configs/infer_finetune.yaml
EXPORT_PRETRAIN_CONFIG ?= configs/export_pretrain.yaml
EXPORT_FINETUNE_CONFIG ?= configs/export_finetune.yaml

TRAIN_CONFIG ?= $(if $(filter pretrain,$(MODEL_STAGE)),$(TRAIN_PRETRAIN_CONFIG),$(TRAIN_FINETUNE_CONFIG))
EVAL_VAL_CONFIG ?= $(if $(filter pretrain,$(MODEL_STAGE)),$(EVAL_VAL_PRETRAIN_CONFIG),$(EVAL_VAL_FINETUNE_CONFIG))
EVAL_TEST_CONFIG ?= $(if $(filter pretrain,$(MODEL_STAGE)),$(EVAL_TEST_PRETRAIN_CONFIG),$(EVAL_TEST_FINETUNE_CONFIG))
INFER_CONFIG ?= $(if $(filter pretrain,$(MODEL_STAGE)),$(INFER_PRETRAIN_CONFIG),$(INFER_FINETUNE_CONFIG))
EXPORT_CONFIG ?= $(if $(filter pretrain,$(MODEL_STAGE)),$(EXPORT_PRETRAIN_CONFIG),$(EXPORT_FINETUNE_CONFIG))
DOCTOR_CONFIG ?= $(TRAIN_PRETRAIN_CONFIG)

TRAIN_ARGS ?=
CURATE_ARGS ?=
SMOKE_TRAIN_ARGS ?=
SMOKE_GATE_ARGS ?=
EVAL_ARGS ?=
INFER_ARGS ?=
EXPORT_ARGS ?=
CONVERSION_ARGS ?=
TEST_ARGS ?=

.PHONY: help env env-update doctor curate-pretrain pretrain-pipeline \
	inspect inspect-all inspect-pretrain inspect-pretrain-smoke inspect-finetune \
	inspect-val inspect-test inspect-val-pretrain inspect-val-finetune \
	inspect-test-pretrain inspect-test-finetune train train-all \
	pretrain finetune train-pretrain train-pretrain-smoke check-pretrain-smoke \
	smoke-pretrain-overfit train-finetune train_pretrain train_finetune \
	eval-val eval-test eval-val-pretrain eval-val-finetune \
	eval-test-pretrain eval-test-finetune eval_val eval_test \
	infer infer-pretrain infer-finetune export export-pretrain export-finetune \
	conversion-datasets conversion-datasets-pretrain conversion-datasets-finetune \
	test compile

define require_model_stage
$(if $(word 2,$(strip $(MODEL_STAGE))),$(error MODEL_STAGE must be exactly pretrain or finetune; got '$(MODEL_STAGE)'),$(if $(filter $(strip $(MODEL_STAGE)),pretrain finetune),,$(error MODEL_STAGE must be pretrain or finetune; got '$(MODEL_STAGE)')))
endef

help:
	@echo Hand Landmarker training system
	@echo   make env             Create the documented Conda environment
	@echo   make env-update      Reconcile and prune the documented environment
	@echo   make doctor          Verify Python, TensorFlow and GPU compatibility
	@echo   make curate-pretrain Materialize the auditable positive-only pretrain snapshot
	@echo   make smoke-pretrain-overfit  Gate full pretrain on a persisted 128-ROI overfit run
	@echo   make pretrain-pipeline Run curation, inspection, smoke gate, then full pretrain
	@echo   make inspect         Validate MODEL_STAGE Train/Val/Test datasets [default: pretrain]
	@echo   make inspect-all     Validate both pretrain and finetune routes
	@echo   make inspect-pretrain Validate stage-1 Train/Val/Test and leakage
	@echo   make inspect-finetune Validate stage-2 Train/Val/Test and leakage
	@echo   make inspect-val     Validate MODEL_STAGE canonical Val ROI set
	@echo   make inspect-test    Validate MODEL_STAGE canonical Test ROI set
	@echo   make train-pretrain  Run stage-1 pseudo-label training
	@echo   make train-finetune  Run stage-2 gold/pseudo fine-tuning
	@echo   make train           Train only MODEL_STAGE [default: pretrain]
	@echo   make train-all       Run both training stages sequentially
	@echo   make eval-val        Evaluate MODEL_STAGE on validation
	@echo   make eval-test       Evaluate MODEL_STAGE on locked test
	@echo   make infer           Run Palm + MODEL_STAGE Hand inference
	@echo   make export          Export and validate MODEL_STAGE ONNX
	@echo   make export EXPORT_ARGS=--force  Bypass only the A1 operator gate for one export
	@echo   make conversion-datasets Build MODEL_STAGE conversion NPY inputs only
	@echo   Set MODEL_STAGE=finetune to route generic targets to stage 2
	@echo   make test            Run unit tests
	@echo   make compile         Compile-check project Python sources

env:
	$(CONDA) env create -f "$(ENV_FILE)"

env-update:
	$(CONDA) env update -f "$(ENV_FILE)" --prune

doctor:
	$(PYTHON) -B scripts/check_environment.py --config "$(DOCTOR_CONFIG)"

curate-pretrain:
	$(PYTHON) -B scripts/curate_pretrain.py --config "$(CURATE_PRETRAIN_CONFIG)" $(CURATE_ARGS)

inspect:
	$(call require_model_stage)
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_CONFIG)"

inspect-all:
	$(MAKE) inspect-pretrain
	$(MAKE) inspect-finetune
	$(MAKE) inspect-val-pretrain
	$(MAKE) inspect-test-pretrain
	$(MAKE) inspect-val-finetune
	$(MAKE) inspect-test-finetune

inspect-pretrain:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_PRETRAIN_CONFIG)"

inspect-pretrain-smoke:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_PRETRAIN_SMOKE_CONFIG)"

inspect-finetune:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_FINETUNE_CONFIG)"

inspect-val:
	$(call require_model_stage)
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_VAL_CONFIG)"

inspect-test:
	$(call require_model_stage)
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_TEST_CONFIG)"

inspect-val-pretrain:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_VAL_PRETRAIN_CONFIG)"

inspect-val-finetune:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_VAL_FINETUNE_CONFIG)"

inspect-test-pretrain:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_TEST_PRETRAIN_CONFIG)"

inspect-test-finetune:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_TEST_FINETUNE_CONFIG)"

train-pretrain: check-pretrain-smoke
	$(PYTHON) -B scripts/train.py --config "$(TRAIN_PRETRAIN_CONFIG)" $(TRAIN_ARGS)

train-pretrain-smoke:
	$(PYTHON) -B scripts/train.py --config "$(TRAIN_PRETRAIN_SMOKE_CONFIG)" $(SMOKE_TRAIN_ARGS)

check-pretrain-smoke:
	$(PYTHON) -B scripts/check_pretrain_smoke.py --config "$(TRAIN_PRETRAIN_SMOKE_CONFIG)" $(SMOKE_GATE_ARGS)

smoke-pretrain-overfit:
	$(MAKE) inspect-pretrain-smoke
	$(MAKE) train-pretrain-smoke
	$(MAKE) check-pretrain-smoke

# Keep the dependent pretrain steps sequential even under `make -j`.
pretrain-pipeline:
	$(MAKE) curate-pretrain
	$(MAKE) inspect-pretrain
	$(MAKE) smoke-pretrain-overfit
	$(MAKE) train-pretrain

train-finetune:
	$(PYTHON) -B scripts/train.py --config "$(TRAIN_FINETUNE_CONFIG)" $(TRAIN_ARGS)

pretrain: train-pretrain

finetune: train-finetune

train:
	$(call require_model_stage)
	$(MAKE) train-$(MODEL_STAGE)

# Keep the two dependent stages sequential even when the caller normally uses
# parallel Make jobs.
train-all:
	$(MAKE) train-pretrain
	$(MAKE) train-finetune

train_pretrain: train-pretrain

train_finetune: train-finetune

eval-val:
	$(call require_model_stage)
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_CONFIG)" $(EVAL_ARGS)

eval-test:
	$(call require_model_stage)
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_CONFIG)" $(EVAL_ARGS)

eval-val-pretrain:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_PRETRAIN_CONFIG)" $(EVAL_ARGS)

eval-val-finetune:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_FINETUNE_CONFIG)" $(EVAL_ARGS)

eval-test-pretrain:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_PRETRAIN_CONFIG)" $(EVAL_ARGS)

eval-test-finetune:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_FINETUNE_CONFIG)" $(EVAL_ARGS)

eval_val: eval-val

eval_test: eval-test

infer:
	$(call require_model_stage)
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_CONFIG)" $(INFER_ARGS)

infer-pretrain:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_PRETRAIN_CONFIG)" $(INFER_ARGS)

infer-finetune:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_FINETUNE_CONFIG)" $(INFER_ARGS)

export:
	$(call require_model_stage)
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_CONFIG)" $(EXPORT_ARGS)

export-pretrain:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_PRETRAIN_CONFIG)" $(EXPORT_ARGS)

export-finetune:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_FINETUNE_CONFIG)" $(EXPORT_ARGS)

conversion-datasets:
	$(call require_model_stage)
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_CONFIG)" $(CONVERSION_ARGS)

conversion-datasets-pretrain:
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_PRETRAIN_CONFIG)" $(CONVERSION_ARGS)

conversion-datasets-finetune:
	$(PYTHON) -B scripts/build_conversion_datasets.py --config "$(EXPORT_FINETUNE_CONFIG)" $(CONVERSION_ARGS)

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
