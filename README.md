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

Install the virtual environment, activate it and install all dependencies:

```bash
$ source venv.sh
```

Define environment variables into `.env` file. Currently, the application requires these environment variables:

- ACCURITY_PROD_DB_SERVER_URI (see [SQLAlchemy engines documentation](https://docs.sqlalchemy.org/en/14/core/engines.html#postgresql) 
for details how to define it)


```bash
$ make env
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
