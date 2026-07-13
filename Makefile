.DEFAULT_GOAL := help

PYTHON ?= python
CONDA ?= conda
ENV_FILE ?= environment.yml

TRAIN_PRETRAIN_CONFIG ?= configs/train_pretrain.yaml
TRAIN_FINETUNE_CONFIG ?= configs/train_finetune.yaml
EVAL_VAL_CONFIG ?= configs/eval_val.yaml
EVAL_TEST_CONFIG ?= configs/eval_test.yaml
INFER_CONFIG ?= configs/infer.yaml
EXPORT_CONFIG ?= configs/export.yaml
DOCTOR_CONFIG ?= $(TRAIN_PRETRAIN_CONFIG)

TRAIN_ARGS ?=
EVAL_ARGS ?=
INFER_ARGS ?=
EXPORT_ARGS ?=
TEST_ARGS ?=

.PHONY: help env env-update doctor inspect inspect-pretrain inspect-finetune \
	inspect-val inspect-test train train-pretrain train-finetune \
	train_pretrain train_finetune eval-val eval-test eval_val eval_test \
	infer export test compile

help:
	@echo Hand Landmarker training system
	@echo   make env             Create the documented Conda environment
	@echo   make env-update      Reconcile and prune the documented environment
	@echo   make doctor          Verify Python, TensorFlow and GPU compatibility
	@echo   make inspect         Validate all canonical Train/Val/Test datasets
	@echo   make inspect-pretrain Validate stage-1 Train/Val/Test and leakage
	@echo   make inspect-finetune Validate stage-2 Train/Val/Test and leakage
	@echo   make inspect-val     Validate the canonical Val ROI set
	@echo   make inspect-test    Validate the canonical Test ROI set
	@echo   make train-pretrain  Run stage-1 pseudo-label training
	@echo   make train-finetune  Run stage-2 gold/pseudo fine-tuning
	@echo   make train           Run both training stages sequentially
	@echo   make eval-val        Evaluate and tune only on the validation set
	@echo   make eval-test       Evaluate the frozen checkpoint on the test set
	@echo   make infer           Run Palm + Hand inference on an image folder
	@echo   make export          Export and validate the ONNX model
	@echo   make test            Run unit tests
	@echo   make compile         Compile-check project Python sources

env:
	$(CONDA) env create -f "$(ENV_FILE)"

env-update:
	$(CONDA) env update -f "$(ENV_FILE)" --prune

doctor:
	$(PYTHON) -B scripts/check_environment.py --config "$(DOCTOR_CONFIG)"

inspect:
	$(MAKE) inspect-pretrain
	$(MAKE) inspect-finetune
	$(MAKE) inspect-val
	$(MAKE) inspect-test

inspect-pretrain:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_PRETRAIN_CONFIG)"

inspect-finetune:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(TRAIN_FINETUNE_CONFIG)"

inspect-val:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_VAL_CONFIG)"

inspect-test:
	$(PYTHON) -B scripts/inspect_dataset.py --config "$(EVAL_TEST_CONFIG)"

train-pretrain:
	$(PYTHON) -B scripts/train.py --config "$(TRAIN_PRETRAIN_CONFIG)" $(TRAIN_ARGS)

train-finetune:
	$(PYTHON) -B scripts/train.py --config "$(TRAIN_FINETUNE_CONFIG)" $(TRAIN_ARGS)

# Keep the two dependent stages sequential even when the caller normally uses
# parallel Make jobs.
train:
	$(MAKE) train-pretrain
	$(MAKE) train-finetune

train_pretrain: train-pretrain

train_finetune: train-finetune

eval-val:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_VAL_CONFIG)" $(EVAL_ARGS)

eval-test:
	$(PYTHON) -B scripts/evaluate.py --config "$(EVAL_TEST_CONFIG)" $(EVAL_ARGS)

eval_val: eval-val

eval_test: eval-test

infer:
	$(PYTHON) -B scripts/infer_folder.py --config "$(INFER_CONFIG)" $(INFER_ARGS)

export:
	$(PYTHON) -B scripts/export_onnx.py --config "$(EXPORT_CONFIG)" $(EXPORT_ARGS)

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
