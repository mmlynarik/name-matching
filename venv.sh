python3.9 -m venv .venv
echo "set -a && . .env && set +a" >> .venv/bin/activate
source .venv/bin/activate
poetry install
