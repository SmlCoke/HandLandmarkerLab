.DEFAULT_GOAL := help

PYTHON ?= python
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT ?= /root/autodl-tmp/TrainFab/HLML-4.0
HLML_SNAPSHOT_ID ?= v4-r1
HLML_EXPERIMENT_ID ?= v4-r1
HLML_RELEASE_ID ?= v4-r1
HLML_STAGE ?= geometry
HLMF_REPO ?= ../../datasets/HandLandmarkerFab
DATASETS_CONFIG ?= configs/datasets.yaml
TRAINING_CONFIG ?= configs/training.yaml
EVALUATION_CONFIG ?= configs/evaluation.yaml
INFERENCE_CONFIG ?= configs/inference.yaml
DEPLOY_CONFIG ?= configs/deploy.yaml

export HAND_DATASET_ROOT HAND_TRAIN_ROOT HLML_SNAPSHOT_ID HLML_EXPERIMENT_ID HLML_RELEASE_ID HLML_STAGE

CLI := $(PYTHON) -B scripts/hlml.py --datasets-config "$(DATASETS_CONFIG)" --training-config "$(TRAINING_CONFIG)" --evaluation-config "$(EVALUATION_CONFIG)" --inference-config "$(INFERENCE_CONFIG)" --deploy-config "$(DEPLOY_CONFIG)"
ROOT_ARGS := --dataset-root "$(HAND_DATASET_ROOT)" --train-root "$(HAND_TRAIN_ROOT)" --snapshot-id "$(HLML_SNAPSHOT_ID)"
DATA_ARGS ?=
TRAIN_ARGS ?=
MINING_ARGS ?=
EVAL_ARGS ?=
FREEZE_ARGS ?=
TEST_ARGS ?=
INFER_ARGS ?=
EXPORT_ARGS ?=

.PHONY: help paths data-audit geometry multitask mine-hard multi-finetune val freeze-winner locked-test infer export environment-check config-check acceptance-smoke test test-unit compile

help:
	@echo HLML 4.0 - HLMF warehouse IDs, zero-copy snapshots, fixed-ROI evaluation
	@echo   make data-audit HLML_STAGE=geometry^|multitask^|multi_finetune
	@echo   make geometry              Audit and train positive-only geometry
	@echo   make multitask             Audit and train from geometry plus published true negatives
	@echo   make mine-hard             Rank Train capture sources and emit an HLMF review request
	@echo   make multi-finetune        Train hard positives plus mandatory pretrain replay
	@echo   make val                   Evaluate fixed reviewed Val Hand ROIs only
	@echo   make freeze-winner         Freeze the only Val-selected winner descriptor
	@echo   make locked-test           Evaluate that winner once on fixed reviewed Test ROIs
	@echo   make infer                 Folder inference; Palm is used only to generate Hand ROIs
	@echo   make export                Export v2 ONNX and enforce A1 operator/numeric contracts
	@echo   make environment-check     Check the server environment
	@echo   make config-check          Parse all five single-purpose public configs
	@echo   make acceptance-smoke      Run HLMF contracts plus synthetic three-stage/fixed-ROI acceptance
	@echo   make test                  Run complete unit tests
	@echo   make compile               Syntax-check Python sources
	@echo Evaluation never runs Palm and does not report Palm misses or original-image cascade metrics.

paths:
	@echo HAND_DATASET_ROOT=$(HAND_DATASET_ROOT)
	@echo HAND_TRAIN_ROOT=$(HAND_TRAIN_ROOT)
	@echo HLML_SNAPSHOT_ID=$(HLML_SNAPSHOT_ID)
	@echo HLML_EXPERIMENT_ID=$(HLML_EXPERIMENT_ID)
	@echo HLML_RELEASE_ID=$(HLML_RELEASE_ID)

data-audit:
	$(CLI) $(ROOT_ARGS) data-audit --stage "$(HLML_STAGE)" $(DATA_ARGS)

geometry:
	$(CLI) $(ROOT_ARGS) data-audit --stage geometry $(DATA_ARGS)
	$(CLI) $(ROOT_ARGS) train --stage geometry $(TRAIN_ARGS)

multitask:
	$(CLI) $(ROOT_ARGS) data-audit --stage multitask $(DATA_ARGS)
	$(CLI) $(ROOT_ARGS) train --stage multitask $(TRAIN_ARGS)

mine-hard:
	$(CLI) $(ROOT_ARGS) mine-hard $(MINING_ARGS)

multi-finetune:
	$(CLI) $(ROOT_ARGS) data-audit --stage multi_finetune $(DATA_ARGS)
	$(CLI) $(ROOT_ARGS) train --stage multi_finetune $(TRAIN_ARGS)

val:
	$(CLI) $(ROOT_ARGS) eval-val $(EVAL_ARGS)

freeze-winner:
	$(CLI) $(ROOT_ARGS) freeze-winner --stage "$(HLML_STAGE)" --release-id "$(HLML_RELEASE_ID)" $(FREEZE_ARGS)

locked-test:
	$(CLI) $(ROOT_ARGS) eval-test --release-id "$(HLML_RELEASE_ID)" $(TEST_ARGS)

infer:
	$(CLI) $(ROOT_ARGS) infer $(INFER_ARGS)

export:
	$(CLI) $(ROOT_ARGS) export $(EXPORT_ARGS)

environment-check:
	$(CLI) $(ROOT_ARGS) environment-check

config-check:
	$(CLI) $(ROOT_ARGS) config-check

acceptance-smoke:
	$(MAKE) -C "$(HLMF_REPO)" compile test
	$(PYTHON) -B -m unittest tests.test_warehouse_v4
	$(CLI) $(ROOT_ARGS) config-check

test test-unit:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py" $(TEST_ARGS)

compile:
	$(PYTHON) -B -c "from pathlib import Path; roots=[Path(value) for value in ('hand_landmarker','models','scripts','tests') if Path(value).exists()]; files=[path for root in roots for path in root.rglob('*.py')]; [compile(path.read_bytes(), str(path), 'exec') for path in files]; print('syntax-checked {} Python files'.format(len(files)))"
