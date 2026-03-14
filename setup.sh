#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# Local K8s dev setup for Salus (uses kind)
# Usage:
#   ./setup.sh          create kind cluster + apply manifests
#   ./setup.sh up       same as above
#   ./setup.sh down     delete all resources (PVCs are kept)
#   ./setup.sh destroy  tear down the entire kind cluster
# ──────────────────────────────────────────────

CLUSTER_NAME="salus"
KIND_CONFIG="kind-config.yaml"
MODE="${1:-up}"

# ── helpers ──────────────────────────────────

log()  { echo "[setup] $*"; }
err()  { echo "[setup] ERROR: $*" >&2; }
dots() { printf "."; }

# wait_for <label> <timeout_seconds> <command...>
# Retries <command> every 3s until it succeeds or times out.
wait_for() {
  local label="$1"
  local timeout="$2"
  shift 2
  local cmd=("$@")
  local elapsed=0

  log "Waiting for: $label"
  while ! "${cmd[@]}" &>/dev/null; do
    if (( elapsed >= timeout )); then
      echo ""
      err "Timed out after ${timeout}s waiting for: $label"
      exit 1
    fi
    dots
    sleep 3
    (( elapsed += 3 ))
  done
  echo ""
  log "Ready: $label"
}

# ── prerequisites ─────────────────────────────

check_prereqs() {
  local missing=()
  command -v kind    &>/dev/null || missing+=("kind")
  command -v kubectl &>/dev/null || missing+=("kubectl")
  if (( ${#missing[@]} > 0 )); then
    err "Missing required tools: ${missing[*]}"
    err "Install via: brew install kind kubectl"
    exit 1
  fi
}

# ── kind cluster lifecycle ────────────────────

ensure_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' already exists — skipping create"
  else
    log "Creating kind cluster '$CLUSTER_NAME'..."
    kind create cluster --name "$CLUSTER_NAME" --config "$KIND_CONFIG"
  fi
  # Point kubectl at this cluster
  kubectl config use-context "kind-${CLUSTER_NAME}"
}

# ── up ────────────────────────────────────────

up() {
  check_prereqs
  ensure_cluster

  log "Applying manifests..."
  kubectl apply -f k8s/kafka.yaml
  kubectl apply -f k8s/timescaledb.yaml
  kubectl apply -f k8s/redis.yaml

  log "Waiting for rollouts..."
  kubectl rollout status statefulset/kafka       --timeout=180s
  kubectl rollout status statefulset/timescaledb --timeout=90s
  kubectl rollout status statefulset/redis       --timeout=60s

  log "Probing Kafka broker (kafka-0)..."
  wait_for "kafka broker on kafka-0" 120 \
    kubectl exec kafka-0 -- \
      kafka-topics.sh --bootstrap-server localhost:9092 --list

  log ""
  log "All services are ready."
  log ""
  log "  ┌─────────────────────────────────────────────┐"
  log "  │  Service        Host                   Port │"
  log "  │  ──────────     ─────────────────────  ──── │"
  log "  │  Kafka          kafka.default.svc       9092 │"
  log "  │  TimescaleDB    timescaledb.default.svc 5432 │"
  log "  │  Redis          redis.default.svc       6379 │"
  log "  └─────────────────────────────────────────────┘"
}

# ── down ──────────────────────────────────────

down() {
  check_prereqs

  if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' does not exist — nothing to do"
    return
  fi

  kubectl config use-context "kind-${CLUSTER_NAME}"

  log "Deleting resources..."
  kubectl delete -f k8s/kafka.yaml       --ignore-not-found
  kubectl delete -f k8s/timescaledb.yaml --ignore-not-found
  kubectl delete -f k8s/redis.yaml       --ignore-not-found

  log ""
  log "Resources deleted."
  log "PersistentVolumeClaims are intentionally kept."
  log "To remove them:  kubectl delete pvc --all"
  log "To destroy the cluster: ./setup.sh destroy"
}

# ── destroy ───────────────────────────────────

destroy() {
  check_prereqs

  if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' does not exist — nothing to do"
    return
  fi

  log "Destroying kind cluster '$CLUSTER_NAME'..."
  kind delete cluster --name "$CLUSTER_NAME"
  log "Cluster deleted. (Docker volumes were removed with it.)"
}

# ── dispatch ──────────────────────────────────

case "$MODE" in
  up)      up      ;;
  down)    down    ;;
  destroy) destroy ;;
  *)
    err "Unknown mode: $MODE  (use: up | down | destroy)"
    exit 1
    ;;
esac
