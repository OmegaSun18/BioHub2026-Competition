# Run all commands from inside the minio/ directory

MINIO_IMAGE      := biohub-minio:latest
MINIO_CONTAINER  := biohub-minio
TEST_IMAGE       := minio-client-test:latest

MINIO_DATA_DIR   := /mnt/data/minio_data
MINIO_ROOT_USER  := biohubadmin
MINIO_ROOT_PASS  := biohub-minio-secret-key-2026
MINIO_BUCKET     := model-registry

.PHONY: all build-minio run-minio wait-minio build-test run-test clean-minio clean-all

# Full end-to-end workflow: build -> start MinIO -> wait -> build & run test -> delete MinIO container
all: build-minio run-minio wait-minio build-test run-test clean-minio

# 1. Build the MinIO repository image (uses minio/Dockerfile)
build-minio:
	docker build -t $(MINIO_IMAGE) .

# 2. Prepare persistent storage on /mnt/data and start the MinIO container
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

# Wait until MinIO is live and startup ingestion finishes before running the test client
wait-minio:
	@echo "Waiting for MinIO health endpoint and model ingestion..."
	@until curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null; do sleep 1; done
	@until docker logs $(MINIO_CONTAINER) 2>&1 | grep -q "Ingestion complete"; do sleep 1; done
	@echo "MinIO is ready and models are ingested."

# 3. Build the test client container from ./minio_test
build-test:
	docker build -t $(TEST_IMAGE) ./minio_test

# 4. Run the test container connecting via 172.17.0.1:9000 (auto-removed via --rm)
run-test:
	docker run --rm \
		-e MINIO_ENDPOINT=172.17.0.1:9000 \
		-e MINIO_ACCESS_KEY=$(MINIO_ROOT_USER) \
		-e MINIO_SECRET_KEY=$(MINIO_ROOT_PASS) \
		$(TEST_IMAGE)

# 5. Stop and remove the MinIO container as a cleanup operation
clean-minio:
	-docker stop $(MINIO_CONTAINER)
	-docker rm $(MINIO_CONTAINER)

# Optional: also remove both built Docker images
clean-all: clean-minio
	-docker rmi $(TEST_IMAGE) $(MINIO_IMAGE)