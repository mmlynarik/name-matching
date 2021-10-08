pip3 install --user virtualenv
virtualenv .venv
echo "set -a && . .env && set +a" >> .venv/bin/activate
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install -e .
