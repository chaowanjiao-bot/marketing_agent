#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROJECT_DIR=$(cd "$DEPLOY_DIR/.." && pwd)
ENV_FILE=${ENV_FILE:-$DEPLOY_DIR/production.env}; OUTPUT_DIR=${OUTPUT_DIR:-$DEPLOY_DIR/rendered}; mkdir -p "$OUTPUT_DIR"
for unit in marketing-agent-api marketing-agent-worker; do sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" -e "s|__ENV_FILE__|$ENV_FILE|g" "$DEPLOY_DIR/$unit.service.in" >"$OUTPUT_DIR/$unit.service"; done
echo "Rendered units in $OUTPUT_DIR. Review them before copying to /etc/systemd/system."
