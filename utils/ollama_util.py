import ollama
from typing import List, Dict

class Ollama:

    model_name = None

    def __init__(self, model_name):
        self.model_name = str(model_name)

    @staticmethod
    def list() -> List[str]:
        return [ model["model"] for model in ollama.list()["models"] ]
    

    def pull(self) -> str:
        if isinstance(self.model_name, str):
            response = ollama.pull(self.model_name)
            return response["status"]

        return "OllamaError: Cannot pull model. Please provide a valid model name."
    

    def exists(self) -> bool:
        # Ollama stores models with a tag (e.g. "model:latest"), so match
        # tagless names against their ":latest" variant too.
        if not isinstance(self.model_name, str):
            return False
    
        name = self.model_name if ":" in self.model_name else f"{self.model_name}:latest"
        return name in Ollama.list()
    

    def delete(self) -> str:
        if isinstance(self.model_name, str):
            response = ollama.delete(self.model_name)
            return response["status"]
        return "OllamaError: Cannot delete model. Please provide a valid model name."
    
    def chat(self, messages: List[Dict[str, str]]) -> ollama.ChatResponse | None:
        if isinstance(self.model_name, str):
            response = ollama.chat(model=self.model_name, messages=messages)
            return response
        return None