from dotenv import load_dotenv
import os
from pydantic import BaseModel, ConfigDict
from typing import List, Dict

class EnvProp(BaseModel):
    model_config = ConfigDict(extra="allow")

class Env():

    def __init__(self) -> None:
        load_dotenv(override=True)
        
        env_keys:List[str] = [key for key, _ in os.environ.items()]
        env_dict:Dict = {key: os.getenv(key) for key in env_keys}
        self.env = EnvProp(**env_dict)

    def get_env(self) -> EnvProp:
        return self.env
