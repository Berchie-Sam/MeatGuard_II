# MeatGuard II pipeline. Every target calls src/main.py, which reads configs/config.yaml.
#
#   make                 # teacher -> baselines -> scratch -> distill -> compress -> export -> evaluate -> xai -> collect
#   make distill         # one step (and the steps it needs, if their results are missing)
#   make smoke           # whole pipeline with three batches per step, results in $(RUNS)_smoke/
#   make PYTHON=D:/RGT/Code/rgtenv/Scripts/python.exe RUNS=runs_local
#
# Each step's target is the summary file it writes last, so make skips steps that have finished. Training
# steps also resume from their last checkpoint when interrupted. Data preparation (sessions, segment) is
# run once, explicitly.

PYTHON ?= python
CONFIG ?= configs/config.yaml
RUNS   ?= runs
MAIN    = $(PYTHON) -m src.main --config $(CONFIG) --runs $(RUNS)

TEACHER   = $(RUNS)/teacher/teacher_selection.json
SCRATCH   = $(RUNS)/student/scratch/summary.json
DISTILLED = $(RUNS)/student/distilled/summary.json
PRUNED    = $(RUNS)/student/pruned/summary.json
EXPORTED  = $(RUNS)/student/export/summary.json
RESULTS   = $(RUNS)/results/test_results.json
XAI       = $(RUNS)/xai/xai_summary.json

.PHONY: all data sessions segment teacher baselines scratch distill compress export evaluate xai collect \
        smoke dry-run setup clean help

all: collect

help:
	$(PYTHON) -m src.main --help

setup:
	$(PYTHON) -m pip install -r requirements.txt

# ---- data preparation (local, once) --------------------------------------------------------------
data: sessions segment
sessions:
	$(MAIN) --mode sessions
segment:
	$(MAIN) --mode segment

# ---- training -------------------------------------------------------------------------------------
teacher: $(TEACHER)
$(TEACHER):
	$(MAIN) --mode teacher

# baselines re-run quickly when finished (each run resumes, sees it is complete, and stops)
baselines: $(TEACHER)
	$(MAIN) --mode baselines

scratch: $(SCRATCH)
$(SCRATCH):
	$(MAIN) --mode scratch

distill: $(DISTILLED)
$(DISTILLED): $(TEACHER)
	$(MAIN) --mode distill

compress: $(PRUNED)
$(PRUNED): $(DISTILLED)
	$(MAIN) --mode compress

export: $(EXPORTED)
$(EXPORTED): $(PRUNED)
	$(MAIN) --mode export

# ---- evaluation -----------------------------------------------------------------------------------
evaluate: $(RESULTS)
$(RESULTS): $(EXPORTED) $(SCRATCH) | baselines
	$(MAIN) --mode evaluate

xai: $(XAI)
$(XAI): $(PRUNED)
	$(MAIN) --mode xai

collect: $(RESULTS) $(XAI)
	$(MAIN) --mode collect

# ---- checks ---------------------------------------------------------------------------------------
smoke:
	$(MAIN) --mode all --smoke

dry-run:
	$(MAIN) --mode all --dry-run

# Removes the smoke-test results only; delete $(RUNS)/ by hand if you really want to discard real results.
clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('$(RUNS)_smoke', ignore_errors=True)"
