# Run all commands from inside the minio/ directory

MINIO_IMAGE      := biohub-minio:latest
MINIO_CONTAINER  := biohub-minio
TEST_IMAGE       := minio-client-test:latest

MINIO_DATA_DIR   := /mnt/data/minio_data
MINIO_ROOT_USER  := biohubadmin
MINIO_ROOT_PASS  := biohub-minio-secret-key-2026
MINIO_BUCKET     := model-registry

.PHONY: all build-minio run-minio ensure-minio wait-minio build-test run-test update-model clean-minio clean-all

# Full end-to-end verification workflow (builds, runs, tests, and removes MinIO container)
all: build-minio run-minio wait-minio build-test run-test clean-minio

# 1. Build the MinIO repository image (uses minio/Dockerfile)
build-minio:
	docker build -t $(MINIO_IMAGE) .

# 2. Prepare persistent storage on /mnt/data and start a fresh MinIO container
run-minio:
	sudo mkdir -p $(MINIO_DATA_DIR)
	sudo chown -R 65532:65532 $(MINIO_DATA_DIR)
	-docker rm -f $(MINIO_CONTAINER) >/dev/null 2>&1 || true
	docker run -d --name $(MINIO_CONTAINER) \
		-p 9000:9000 \
		-p 9001:9001 \
		-e MINIO_ROOT_USER=$(MINIO_ROOT_USER) \
		-e MINIO_ROOT_PASSWORD=$(MINIO_ROOT_PASS) \
		-e MINIO_BUCKET=$(MINIO_BUCKET) \
		-v $(MINIO_DATA_DIR):/data \
		-v /mnt/data:/mnt/data:ro \
		$(MINIO_IMAGE)

# Start MinIO only if it is not already running
ensure-minio:
	@if [ -z "$$(docker ps -q -f name=^/$(MINIO_CONTAINER)$$)" ]; then \
		$(MAKE) run-minio; \
	else \
		echo "$(MINIO_CONTAINER) is already running."; \
	fi

# Wait until MinIO is live and startup ingestion finishes
wait-minio:
	@echo "Waiting for MinIO health endpoint and model ingestion..."
	@until curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null; do sleep 1; done
	@until docker logs $(MINIO_CONTAINER) 2>&1 | grep -q "Ingestion complete"; do sleep 1; done
	@echo "MinIO is ready."

# 3. Build the client container from ./minio_test
build-test:
	docker build -t $(TEST_IMAGE) ./minio_test

# 4. Run the read-only verification test container (auto-removed via --rm)
run-test:
	docker run --rm \
		-e MINIO_ENDPOINT=172.17.0.1:9000 \
		-e MINIO_ACCESS_KEY=$(MINIO_ROOT_USER) \
		-e MINIO_SECRET_KEY=$(MINIO_ROOT_PASS) \
		-e MINIO_BUCKET=$(MINIO_BUCKET) \$(TEST_IMAGE)

# 5. Apply updates defined in minio_test/update_model.py (starts MinIO if stopped)
update-model: ensure-minio wait-minio
	@if [ -z "$$(docker images -q$(TEST_IMAGE))" ]; then \
		$(MAKE) build-test; \
	fi
	docker run --rm \
		-e MINIO_ENDPOINT=172.17.0.1:9000 \
		-e MINIO_ACCESS_KEY=$(MINIO_ROOT_USER) \
		-e MINIO_SECRET_KEY=$(MINIO_ROOT_PASS) \
		-e MINIO_BUCKET=$(MINIO_BUCKET) \
		-v $(CURDIR)/minio_test:/app:ro \
		-v /mnt/data:/mnt/data:ro \
		$(TEST_IMAGE) python /app/update_model.py

# 6. Stop and remove the MinIO container (persisted data in /mnt/data/minio_data remains intact)
clean-minio:
	-docker stop $(MINIO_CONTAINER)
	-docker rm $(MINIO_CONTAINER)

# Optional: remove containers and both built Docker images
clean-all: clean-minio
	-docker rmi $(TEST_IMAGE)$(MINIO_IMAGE)