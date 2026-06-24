from pydantic import BaseModel, ConfigDict

class EnvType(BaseModel):
    model_config = ConfigDict(extra="allow")