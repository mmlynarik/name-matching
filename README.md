Accurity Automated Mappings Data Science Project
==============================

Project Organization
--------------------

```
├── data
│   ├── interim        <- Intermediate data that has been transformed
│   ├── output         <- The output data as a result of the modeling procedures
│   └── raw            <- The original, immutable source data
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks
├── references         <- Manuals, weblinks, papers and all other explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
├── src                <- Python source code for deployment
│   ├── __init__.py    <- Makes src a Python package
│   ├── adapters       <- Object-relational mapper, repository and HTTP responses definitions
|   ├── domain         <- Domain model, i.e. classes related to statistical/ML model
│   ├── entrypoints    <- FastAPI application files
|   ├── service_layer  <- Unit-of-work and service-layer functions definitions
├── tests              <- E2E tests
├── .env_tmpl          <- Environment variables template file
├── .gitignore         <- .gitignore file
├── docker-compose.yml <- Docker-compose file used for running the web app as a containerized application
├── Dockerfile         <- Dockerfile used for building project docker image
├── Makefile           <- Makefile for running `make app` or `make env` commands
├── oracle_client.zip  <- Oracle client necessary for connecting to Oracle databases
├── README.md          <- The top-level README for developers
├── requirements.txt   <- Dependencies file
├── setup.py           <- Makes project pip-installable (using pip install -e .) 
├── venv.sh            <- Virtual environment initialization script  

```

--------


Set up
------------

Run these commands in Windows Powershell to install Ubuntu in WSL:

```powershell
Invoke-WebRequest -Uri https://aka.ms/wslubuntu2004 -OutFile Ubuntu.appx -UseBasicParsing
Add-AppxPackage Ubuntu.appx
```

Run these commands in Ubuntu shell to install Python 3.9, generate SSH keys for GitHub, clone repository and set up VSCode remote server:
```bash
mkdir python
cd python

sudo apt update
sudo apt -y upgrade

sudo apt install -y python3.9
sudo apt install -y python3.9-venv
sudo apt install -y python3.9-dev
sudo apt install -y python-is-python3
sudo apt install -y --reinstall build-essential
sudo apt install -y libgmp-dev portaudio19-dev libssl-dev python3-dev
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.9 -

echo export PATH=\"\$HOME/.poetry/bin:\$PATH\" >> ~/.bashrc
echo alias cl=clear >> ~/.bashrc
echo alias jl=\"jupyter lab --no-browser --port 8888 --ip=\'127.0.0.1\' --ContentManager.allow_hidden=True --ServerApp.token=\'\' --ServerApp.password=\'\'\" >> ~/.bashrc


ssh-keygen -t rsa -b 4096 -C "name.surname@gmail.com"
# copy SSH public key to your github account

git config --global user.name "Name Surname"
git config --global user.email "name.surname@gmail.com"
git clone git@github.com:mmlynarik/name-matching.git

cd name-matching/
code .
```

Define environment variables into `.env` file. Currently, the application requires these environment variables:

- ACCURITY_PROD_DB_SERVER_URI (see [SQLAlchemy engines documentation](https://docs.sqlalchemy.org/en/14/core/engines.html#postgresql) 
for details how to define it)


```bash
$ make env
```


Install the virtual environment, activate it and install all dependencies:

```bash
$ source venv.sh
```

Run the `FastAPI` application using `uvicorn` server:

```bash
$ make app
```

Launch test suite using `pytest` and `coverage`. The prerequisite is the ACCURITY_TEST_DB_SERVER_URI environment variable defined in `.env` file

```bash
$ make test
```

For data science development, run Jupyter Lab and open the notebooks in `notebooks/`:

```bash
$ jupyter lab
```
