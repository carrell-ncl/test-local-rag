
## Setup

Make sure you have the right Ollama models pulled

You already have llama3.1 (from your /api/tags output). You still need the embedding model:
docker compose exec ollama ollama pull nomic-embed-text


## Deploying using Docker Compose file

Deploy services
docker compose up -d

Check running services (containers)
docker compose ps

Stop services
docker compose down 

## Run the program
Ensure models are present
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.1

Start the ingestion worker (Terminal A)
python -m src.ingest_worker --index rag_chunks

(OPTIONAL) Run above using Makerfile
Run command: make worker


Confirm index has data
curl -s "http://localhost:9200/rag_chunks/_count" | cat


Ask a question (Terminal B)
python -m src.query_rag \
  --question "Which paper introduces attention mechanisms and how do they work?" \
  --categories cs.LG,cs.AI,stat.ML \
  --show-top-papers


## Add a single article steps

Deploy services
docker compose up -d

Sanity check
curl http://localhost:9200 | head
curl http://localhost:11434/api/tags

Pull the Ollama embedding model
docker compose exec ollama ollama pull nomic-embed-text

Create the OpenSearch index (one-time)
This:
    asks Ollama for embedding dimension
    creates the index mapping
python src/os_index.py --index rag_chunks --embed-model nomic-embed-text

Varify
curl -s "http://localhost:9200/rag_chunks" | head

Starting the worker
python src/ingest_worker.py --index rag_chunks

Choose an arXiv paper
Example (Graph Neural Networks classic):
    Graph Attention Networks
    arXiv ID: 1710.10903
    PDF: https://arxiv.org/pdf/1710.10903.pdf

job.json for the documents

Enqueue the job into Redis
From a second terminal:
python src/ingest_worker.py --enqueue "$(cat job.json)"

Watch ingestion happen
Back in the worker terminal you should see something like:
[INFO] 1710.10903 -> 20 chunks
[OK] Indexed 20 chunks for 1710.10903

What just happened:
PDF downloaded
Text extracted (references removed)
Split into sections
Chunked (~900 chars)
Embedded via Ollama
Indexed into OpenSearch

Verify the article is indexed
curl -X GET "http://localhost:9200/rag_chunks/_search?size=3" -H "Content-Type: application/json" -d '{
    "query": {
      "term": { "arxiv_id": "1710.10903" }
    }
  }'


### Debugging a service that did not deploy from the compose

Check logs for a service (or service that is not running)
docker logs --tail=200 test-local-rag-opensearch-1


## Running a query

Best default command (no filters)
python -m src.query_rag --question "What is the key idea behind graph attention in GAT?" --show-top-papers

With category filters (recommended for arXiv)
python -m src.query_rag \
  --question "What is oversmoothing in GNNs and how is it mitigated?" \
  --categories cs.LG,cs.AI,stat.ML \
  --show-top-papers

Keep context balanced across papers
python -m src.query_rag \
  --question "Summarise the main contributions across the most relevant papers." \
  --max-docs 4 \
  --max-chunks-per-doc 2 \
  --k 8 \
  --show-top-papers

Force single-paper mode when you want it
python -m src.query_rag \
  --question "Explain the method section." \
  --arxiv-id 1710.10903 \
  --show-context
