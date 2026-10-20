#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v docker >/dev/null || { echo 'Install Docker Engine and Compose v2 first.'; exit 1; }
test -f .env || { echo 'Copy .env.example to .env and configure secrets and domain.'; exit 1; }
if grep -q 'REPLACE_WITH' .env; then echo 'Replace the placeholder secrets first.'; exit 1; fi
docker compose up --build -d
printf 'API/UI available on localhost:8000. Put an HTTPS reverse proxy in front before remote access.\n'
