SHELL := /bin/bash
.DEFAULT_GOAL := help

PY            ?= .venv/bin/python
PY_ML         ?= .venv312/bin/python
DATA          ?= data/cctv_images
SCRAPERS      ?= geocode_cameras.py build_ban_zone.py
WEATHER_TZ    ?= Asia/Bangkok
WEATHER_LABEL ?= end
RAIN_LABEL    ?= start
FORCE         ?=

OUT := data_out
P   := pipeline (เก้อ)
S   := scraping (เพา)
EXT := $(S)/external_data/data
CCTV_INFO   ?= $(EXT)/cameras.csv
RAIN        ?= $(EXT)/rain_hourly.csv
TRUCK_RULES ?= $(EXT)/truck_rules.csv
EXT_STAMP   := $(OUT)/.external_inputs

ARCH := $(shell uname -s)-$(shell uname -m)
ifeq ($(ARCH),Darwin-arm64)
BACKEND ?= coreml
else
BACKEND ?= torch
endif
ifeq ($(BACKEND),coreml)
PY_DET    = $(PY_ML)
DET_MODEL = models/yolo11s_nonms.mlpackage
NEED_CLIP = @test -e models/clip_ViT-B-16.mlpackage || bash models/download.sh
else
PY_DET    = $(PY)
DET_MODEL = models/yolo11s.pt
NEED_CLIP = @true
endif
CLS_MODEL := models/truck_clipvitb.pt

CHECK     = $(PY) "$(P)/validate.py"
NEED_DATA = @test -d "$(DATA)" || bash models/download.sh --data

FEATURES   := $(OUT)/features.csv
CLEAN      := $(OUT)/images_clean.csv
CAMERAS    := $(OUT)/cameras.csv
DETECTIONS := $(OUT)/detections.parquet
CANDIDATES := $(OUT)/candidates.parquet
VEHICLES   := $(OUT)/vehicles.parquet
GEO        := $(OUT)/cameras_geo.csv
ZONE       := config/ban_zone.geojson
RULES      := config/ban_rules.json
WEATHER    := $(wildcard $(OUT)/weather_hourly.csv)
CONTEXT    := $(OUT)/context_camera_hour.csv
EXT_FILES  := "$(CCTV_INFO)" "$(RAIN)" "$(TRUCK_RULES)"
SIGHTINGS  := $(OUT)/heavy_sightings.parquet
AGG        := $(OUT)/agg_camera_hour.csv
CLIP       := $(OUT)/truck_violations.mp4

stage_extract    := $(FEATURES)
stage_cleanse    := $(CLEAN)
stage_detect     := $(DETECTIONS)
stage_candidates := $(CANDIDATES)
stage_classify   := $(VEHICLES)
stage_join       := $(CONTEXT)
stage_rules      := $(SIGHTINGS)
stage_clip       := $(CLIP)

ORDER := $(FEATURES) $(CLEAN) $(CAMERAS) $(DETECTIONS) $(CANDIDATES) $(OUT)/candidates_emb.npy $(VEHICLES) \
         $(GEO) $(ZONE) $(RULES) $(OUT)/weather_hourly.csv $(EXT_STAMP) $(CONTEXT) $(SIGHTINGS) $(AGG) $(CLIP)
PATH_FILES := $(FEATURES) $(CLEAN) $(DETECTIONS) $(CANDIDATES) $(VEHICLES) $(SIGHTINGS) $(OUT)/yoloe_eval.csv \
              "modeling (พัท)/labels/labels.csv"

$(shell for f in $(EXT_FILES); do if [ -e "$$f" ] && [ "$$f" -nt $(EXT_STAMP) ]; then touch $(EXT_STAMP); fi; done)

ifneq ($(shell command -v uv 2>/dev/null),)
VENV = uv venv --allow-existing --python $(1) $(2)
PIP  = uv pip install --python $(1) -r $(2)
else
VENV = python$(1) -m venv $(2)
PIP  = $(1) -m pip install -r $(2)
endif

.PHONY: help all extract cleanse scraped scrape detect candidates classify join rules clip validate flow \
        prefect-ui dashboard evidence setup setup-main setup-coreml lock models data coreml portable touch \
        $(foreach s,$(FORCE),$(stage_$(s)))

help:
	@echo "make setup          .venv (Python 3.14) + .venv312 (Python 3.12, CoreML)"
	@echo "make models         model weights -> models/"
	@echo "make data           raw CCTV images -> $(DATA)"
	@echo "make all            scraped -> extract -> cleanse -> detect -> candidates -> classify -> join -> rules -> validate"
	@echo "make flow           same pipeline as a Prefect flow"
	@echo "make prefect-ui     Prefect UI on http://127.0.0.1:4200"
	@echo "make validate       data-quality checks on every stage output"
	@echo "make <stage>        extract | cleanse | scraped | scrape | detect | candidates | classify | join | rules | clip"
	@echo "make dashboard      Streamlit dashboard"
	@echo "make touch          mark committed outputs up to date after git clone"
	@echo "make portable       image paths -> $(DATA)/<camera>/<file>"
	@echo "make coreml         re-export CoreML packages from yolo11s.pt and open_clip"
	@echo "make lock           regenerate requirements.lock files"
	@echo "BACKEND=$(BACKEND) PY=$(PY) PY_ML=$(PY_ML) DATA=$(DATA) FORCE='$(FORCE)'"

all: $(SIGHTINGS) $(AGG)
	@$(CHECK) --all

extract: $(FEATURES)
cleanse: $(CLEAN) $(CAMERAS)
detect: $(DETECTIONS)
candidates: $(CANDIDATES)
classify: $(VEHICLES)
join: $(CONTEXT)
rules: $(SIGHTINGS) $(AGG)
clip: $(CLIP)

scraped: $(GEO) $(ZONE) $(RULES)
	@$(CHECK) $^ $(WEATHER) $(EXT_FILES)

scrape: | $(CAMERAS)
	@set -e; for s in $(SCRAPERS); do echo "$(PY) \"$(S)/$$s\""; $(PY) "$(S)/$$s"; done
	@$(CHECK) $(GEO) $(ZONE) $(RULES) $(WEATHER) $(EXT_FILES)

validate:
	@$(CHECK) --all

$(FEATURES):
	$(NEED_DATA)
	$(PY) "$(P)/extract_features.py" "$(DATA)" $@
	@$(CHECK) $@

$(CLEAN): $(FEATURES)
	@$(CHECK) $<
	$(PY) "$(P)/clean.py" $< $(OUT)
	@$(CHECK) $@ $(CAMERAS)

$(CAMERAS): $(CLEAN) ;

$(DETECTIONS): $(CLEAN)
	@$(CHECK) $<
	@test -e $(DET_MODEL) || bash models/download.sh
	$(NEED_DATA)
	$(PY_DET) "$(P)/detect.py" $< $@ --backend $(BACKEND) --model $(DET_MODEL)
	@$(CHECK) $@

$(CANDIDATES): $(DETECTIONS)
	@$(CHECK) $<
	$(NEED_CLIP)
	$(NEED_DATA)
	$(PY_DET) "$(P)/classify_trucks.py" embed $< $(OUT)/candidates --backend $(BACKEND)
	@$(CHECK) $@

$(VEHICLES): $(CANDIDATES)
	@$(CHECK) $<
	@test -e $(CLS_MODEL) || bash models/download.sh
	$(NEED_DATA)
	$(PY) "$(P)/classify_vit.py" $< $(CLS_MODEL) $@
	@$(CHECK) $@

$(GEO): | $(CAMERAS)
	$(PY) "$(S)/geocode_cameras.py"
	$(PY) "$(S)/build_ban_zone.py"

$(ZONE): | $(GEO)
	$(PY) "$(S)/build_ban_zone.py"

$(EXT_STAMP):
	@touch $@

$(CONTEXT): $(CLEAN) $(GEO) $(ZONE) $(WEATHER) $(EXT_STAMP)
	@$(CHECK) $^ $(EXT_FILES)
	$(PY) "$(P)/official_rules.py" --rules "$(TRUCK_RULES)"
	$(PY) "$(P)/join_scraped.py" --cameras "$(CCTV_INFO)" --rain "$(RAIN)" --rain-label $(RAIN_LABEL) \
		--weather-tz $(WEATHER_TZ) --label $(WEATHER_LABEL)
	@$(CHECK) $@ $(GEO) $(RULES)

$(SIGHTINGS): $(VEHICLES) $(CLEAN) $(CONTEXT) $(RULES)
	@$(CHECK) $^
	$(PY) "$(P)/rules.py"
	@$(CHECK) $@ $(AGG)

$(AGG): $(SIGHTINGS) ;

$(CLIP): $(SIGHTINGS) $(GEO) $(ZONE) $(RULES)
	$(PY) "dashboard (เต้)/make_timelapse.py" --fps 6 --out $@

flow:
	$(PY) "$(P)/flow.py" --backend $(BACKEND) --py-ml $(PY_ML) --data $(DATA) $(if $(strip $(FORCE)),--force $(FORCE))

prefect-ui:
	$(PY) -m prefect server start

dashboard:
	$(PY) -m streamlit run "dashboard (เต้)/app.py"

evidence:
	$(PY) "insights (ม่อน)/make_evidence.py"

setup: setup-main setup-coreml

setup-main:
	$(call VENV,3.14,.venv)
	$(call PIP,$(PY),requirements.lock)

setup-coreml:
ifeq ($(ARCH),Darwin-arm64)
	$(call VENV,3.12,.venv312)
	$(call PIP,$(PY_ML),"$(P)/requirements-coreml.lock")
else
	@echo "skip .venv312: CoreML needs macOS on Apple Silicon, use BACKEND=torch"
endif

lock:
	MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements.txt --python-version 3.14 \
		--python-platform aarch64-apple-darwin --only-binary :all: -o requirements.lock
	MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile "$(P)/requirements-coreml.txt" --python-version 3.12 \
		--python-platform aarch64-apple-darwin --only-binary :all: -o "$(P)/requirements-coreml.lock"

models:
	bash models/download.sh

data:
	bash models/download.sh --data

coreml:
	@test -e models/yolo11s.pt || bash models/download.sh
	$(PY_ML) "$(P)/export_coreml.py"

portable:
	$(PY) "$(P)/portable_paths.py" $(PATH_FILES) --root $(DATA)

touch:
	@for f in $(EXT_FILES); do if [ -e "$$f" ]; then touch -t 202601010000.00 "$$f"; fi; done
	@i=10; for f in $(ORDER); do if [ -e "$$f" ]; then touch -t 202601010000.$$i "$$f"; echo "touch $$f"; fi; \
		i=$$((i+1)); done
