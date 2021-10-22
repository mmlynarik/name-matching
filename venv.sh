python3 -m venv .venv
echo "set -a && . .env && set +a" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
