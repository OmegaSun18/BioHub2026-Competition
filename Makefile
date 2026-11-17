SHELL := /bin/bash
export KUBECONFIG := /etc/rancher/k3s/k3s.yaml

IMAGE_NAME     ?= biohub-pipeline:latest
TARGET_QPS     ?= 4.0
TEST_DATA_DIR  ?= /mnt/data/biohub_working/split_dataset/test
DET_REPLICAS   ?= 2
TRACK_REPLICAS ?= 2

.DEFAULT_GOAL := help
.PHONY: help setup-fixes build deploy scale monitor load-gen run-all status clean-k8s clean-docker clean

help: ## Display this help message with all available targets
	@echo "BioHub 2026 Pipeline Management Targets:"
	@echo "----------------------------------------"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup-fixes: ## Persist and apply all K3s, networking, DNS, and YAML fixes idempotently
	@command -v socat >/dev/null 2>&1 || (sudo apt-get update -qq && sudo apt-get install -y -qq socat)
	@sudo modprobe br_netfilter || true
	@printf "net.bridge.bridge-nf-call-iptables = 1\nnet.ipv4.ip_forward = 1\n" | sudo tee /etc/sysctl.d/99-k3s.conf >/dev/null
	@sudo sysctl --system >/dev/null
	@sudo iptables -P FORWARD ACCEPT
	@sudo iptables -C FORWARD -i cni0 -j ACCEPT 2>/dev/null || sudo iptables -I FORWARD 1 -i cni0 -j ACCEPT
	@sudo iptables -C FORWARD -o cni0 -j ACCEPT 2>/dev/null || sudo iptables -I FORWARD 1 -o cni0 -j ACCEPT
	@sudo mkdir -p /etc/systemd/system/k3s.service.d
	@printf "[Service]\nExecStartPost=/sbin/iptables -P FORWARD ACCEPT\nExecStartPost=/sbin/iptables -I FORWARD 1 -i cni0 -j ACCEPT\nExecStartPost=/sbin/iptables -I FORWARD 1 -o cni0 -j ACCEPT\n" | sudo tee /etc/systemd/system/k3s.service.d/cni-firewall.conf >/dev/null
	@sudo systemctl daemon-reload
	@sudo mkdir -p /mnt/data/k3s /etc/rancher/k3s
	@NEW_CFG=$$(printf 'data-dir: "/mnt/data/k3s"\ndocker: true\nwrite-kubeconfig-mode: "0644"\ndisable-network-policy: true\ndisable:\n  - traefik\n'); \
	CUR_CFG=$$(sudo cat /etc/rancher/k3s/config.yaml 2>/dev/null || true); \
	if [ "$$NEW_CFG" != "$$CUR_CFG" ]; then \
	    echo "$$NEW_CFG" | sudo tee /etc/rancher/k3s/config.yaml >/dev/null; \
	    sudo systemctl restart k3s; \
	    sleep 5; \
	fi
	@kubectl wait --for=condition=Ready nodes --all --timeout=60s >/dev/null
	@if [ -f k8s/pipeline.yaml ]; then sed -i 's/clusterIP: None/type: ClusterIP/g' k8s/pipeline.yaml; fi
	@echo "All host and K3s fixes verified."

build: ## Build the Docker image (compiles pipeline.proto and packages server/load_gen)
	docker build -t $(IMAGE_NAME) .

deploy: setup-fixes ## Deploy detection, tracking, Prometheus, and Grafana to Kubernetes
	kubectl apply -f k8s/pipeline.yaml
	kubectl rollout status deployment/cell-detection --timeout=120s
	kubectl rollout status deployment/cell-tracking --timeout=120s
	kubectl rollout status deployment/prometheus --timeout=120s
	kubectl rollout status deployment/grafana --timeout=120s

scale: ## Scale detection and tracking stages (e.g. make scale DET_REPLICAS=3 TRACK_REPLICAS=2)
	./scale_stage.sh detection $(DET_REPLICAS)
	./scale_stage.sh tracking $(TRACK_REPLICAS)
	kubectl rollout status deployment/cell-detection --timeout=120s
	kubectl rollout status deployment/cell-tracking --timeout=120s

monitor: ## Start Prometheus (9090) & Grafana (3000) port-forwards and verify scrape targets
	@pkill -f "[k]ubectl port-forward" || true
	@nohup kubectl port-forward svc/prometheus 9090:9090 >/tmp/pf-prometheus.log 2>&1 &
	@nohup kubectl port-forward svc/grafana 3000:3000 >/tmp/pf-grafana.log 2>&1 &
	@sleep 3
	@curl -s http://127.0.0.1:9090/api/v1/targets | grep -o '"health":"[^"]*"' || echo "Waiting for first scrape..."
	@echo "Prometheus: http://127.0.0.1:9090 | Grafana: http://127.0.0.1:3000"

load-gen: ## Run the open-loop load generator against the deployed Kubernetes services
	@kubectl delete pod load-gen --ignore-not-found --wait=true
	@kubectl run load-gen --image=$(IMAGE_NAME) --image-pull-policy=IfNotPresent \
	    --restart=Never \
	    --overrides='{"spec":{"containers":[{"name":"load-gen","image":"$(IMAGE_NAME)","imagePullPolicy":"IfNotPresent","command":["python","/workspace/load_gen.py"],"env":[{"name":"GRPC_DNS_RESOLVER","value":"native"},{"name":"DETECTION_ADDR","value":"detection-service.default.svc.cluster.local:50051"},{"name":"TRACKING_ADDR","value":"tracking-service.default.svc.cluster.local:50051"},{"name":"TARGET_QPS","value":"$(TARGET_QPS)"},{"name":"TEST_DATA_DIR","value":"$(TEST_DATA_DIR)"}],"volumeMounts":[{"name":"data-vol","mountPath":"/mnt/data","readOnly":true}]}],"volumes":[{"name":"data-vol","hostPath":{"path":"/mnt/data","type":"Directory"}}]}}'
	@kubectl wait --for=condition=Ready pod/load-gen --timeout=60s || true
	@kubectl logs -f load-gen

run-all: deploy monitor load-gen ## Full end-to-end run: apply fixes, deploy cluster, start monitoring, run load-gen

status: ## Check pod, service, and endpoint status across the cluster
	kubectl get pods,svc,endpoints -o wide

clean-k8s: ## Safely remove all Kubernetes pipeline resources and port-forwards
	@pkill -f "[k]ubectl port-forward" || true
	@kubectl delete pod load-gen dns-check net-test --ignore-not-found
	@kubectl delete -f k8s/pipeline.yaml --ignore-not-found
	@kubectl get all

clean-docker: ## Safely remove standalone test containers and dangling image layers
	@docker rm -f det-test track-test cell-detection-test cell-tracking-test 2>/dev/null || true
	@docker image prune -f

clean: clean-k8s clean-docker ## Clean everything started in Kubernetes and Docker without touching /mnt/data
	@echo "Full environment cleanup complete."
