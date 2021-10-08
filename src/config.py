from pydantic import BaseSettings
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.base import float_format


class Settings(BaseSettings):
    ACCURITY_PROD_DB_SERVER_URI: str
    ACCURITY_TEST_DB_SERVER_URI: str = None

    class Config:
        env_file = ".env"

    @property
    def DATABASE_TYPE(self):
        if "postgres" in self.ACCURITY_PROD_DB_SERVER_URI.split("://")[0]:
            return "POSTGRES"
        elif "oracle" in self.ACCURITY_PROD_DB_SERVER_URI.split("://")[0]:
            return "ORACLE"
        elif "mssql" in self.ACCURITY_PROD_DB_SERVER_URI.split("://")[0]:
            return "MSSQL"


class NotebookSettings:
    def __init__(self):
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_colwidth", 100)
        pd.set_option("display.float_format", float_format)

        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.xmargin"] = 0.01
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams['patch.force_edgecolor'] = True
        # plt.style.use("ggplot") plt.style.use('fivethirtyeight')


settings = Settings()
nbsettings = NotebookSettings()
