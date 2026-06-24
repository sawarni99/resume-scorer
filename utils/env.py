from dotenv import load_dotenv
import os
from typing import List, Dict
from datatypes.env_type import EnvType



class Env():

    def __init__(self) -> None:
        load_dotenv(override=True)
        
        env_keys:List[str] = [key for key, _ in os.environ.items()]
        env_dict:Dict = {key: os.getenv(key) for key in env_keys}
        self.env = EnvType(**env_dict)

    def get_env(self) -> EnvType:
        return self.env
