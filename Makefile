# openjev: one-pass option scoring with Gemma 3 4B via MLX. Native macOS only (Metal).
VENV   := .venv
BIN    := $(VENV)/bin
MODEL  ?= models/gemma-3-4b-it
HF_REPO ?= google/gemma-3-4b-it
HOST   ?= 127.0.0.1
PORT   ?= 8000
NORM   ?= mean
DATA   ?= data/synthetic/test.jsonl
API_KEY ?= local
SCENARIO ?= defend_the_center
API    ?= score
DATA_DIR ?= data/synthetic
FEATS  ?= runs/feats
HEAD   ?= runs/head.safetensors
SEP    ?= \nChoice: 

.PHONY: help setup venv model serve health request doom systemone score check bench eval features train eval-head clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: venv model ## create the venv and download the model

venv: ## uv sync with the torch extra (jevlike comparison / HF cross-checks)
	uv sync --extra torch

model: ## download $(HF_REPO) into $(MODEL) if missing
	@test -f $(MODEL)/config.json || hf download $(HF_REPO) --local-dir $(MODEL)

serve: ## run the HTTP server (loads the model once; POST /score, POST /v1/systemone, GET /health)
	$(BIN)/openjev serve --host $(HOST) --port $(PORT) --model $(MODEL)

health: ## curl the running server's health endpoint
	@curl -s $(HOST):$(PORT)/health; echo

request: ## example scoring request against the running server
	@curl -s $(HOST):$(PORT)/score -H 'content-type: application/json' -d '{"context": "Customer: my order arrived broken. Agent:", "options": [" I am sorry to hear that, I will send a replacement today.", " Please read our returns policy.", " Have you tried turning it off and on again?"], "norm": "$(NORM)"}'; echo

doom: ## play Doom in the terminal, the running server picks every action (SCENARIO=defend_the_center API=score)
	$(BIN)/python demo/doom/play.py --scenario $(SCENARIO) --api $(API) --url http://$(HOST):$(PORT) --api-key $(API_KEY)

systemone: ## TypeSafe quickstart example (choice + score + noul) against the running server
	@curl -s $(HOST):$(PORT)/v1/systemone -H 'Authorization: Bearer $(API_KEY)' -H 'Content-Type: application/json' -d @examples/systemone-quickstart.json; echo

score: ## one-off CLI scoring, e.g. make score CONTEXT="..." OPTIONS="--option a --option b"
	$(BIN)/openjev score --model $(MODEL) --norm $(NORM) --context "$(CONTEXT)" $(OPTIONS)

check: ## verify prefix-cached batched scores match naive re-encoding
	$(BIN)/openjev check --model $(MODEL)

bench: ## latency: cached+batched vs naive, 200-token context x 8 options
	$(BIN)/openjev bench --model $(MODEL)

eval: ## zero-shot top-k accuracy on jevlike-style JSONL ($(DATA))
	$(BIN)/openjev eval $(DATA) --model $(MODEL) --norm $(NORM)

features: ## cache frozen-Gemma features for $(DATA_DIR)/{train,validation,test}.jsonl into $(FEATS)/
	@for split in train validation test; do \
	  $(BIN)/openjev features $(DATA_DIR)/$$split.jsonl --out $(FEATS)/$$split.npz --model $(MODEL) --sep "$$(printf '$(SEP)')"; \
	done

train: ## train the attention head on $(FEATS)/train.npz, validate on $(FEATS)/validation.npz -> $(HEAD)
	$(BIN)/openjev train $(FEATS)/train.npz --validation $(FEATS)/validation.npz --out $(HEAD)

eval-head: ## top-k, ECE and shuffled-context control of $(HEAD) on $(FEATS)/test.npz
	$(BIN)/openjev eval-head $(HEAD) $(FEATS)/test.npz

clean: ## remove caches and run artefacts (keeps the venv and model)
	rm -rf runs __pycache__ openjev/__pycache__
