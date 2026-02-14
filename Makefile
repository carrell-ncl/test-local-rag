SHELL := /bin/bash
ENV_NAME := test_rag

CHUNKS_INDEX := rag_chunks
DOCS_INDEX := rag_docs
EMBED_MODEL := nomic-embed-text

.PHONY: worker ensure-docs-index build-docs rebuild-docs

worker:
	@source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	python -m src.ingest_worker --index $(CHUNKS_INDEX)

ui:
	@source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	streamlit run app/streamlit_app.py

ensure-docs-index:
	@source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	python -m src.os_index_docs --index $(DOCS_INDEX) --embed-model $(EMBED_MODEL)

build-docs:
	@source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	python -m src.build_rag_docs \
		--chunks-index $(CHUNKS_INDEX) \
		--docs-index $(DOCS_INDEX) \
		--embed-model $(EMBED_MODEL)

rebuild-docs:
	@curl -s -X DELETE "http://localhost:9200/$(DOCS_INDEX)" >/dev/null || true
	@$(MAKE) ensure-docs-index
	@$(MAKE) build-docs
