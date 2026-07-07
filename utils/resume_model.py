
from utils.utility import Utility
from utils.constants import Constants
from typing import List, Dict, Any
from utils.ollama_util import Ollama
from utils.resume_util import ResumeUtil
import evaluate
import json

class ResumeModel:
    model_name = None
    utility = Utility()

    def __init__(self, model_name):
        self.model_name = str(model_name)
        self.rouge_metrics = evaluate.load("rouge")
        self.exact_match_metrics = evaluate.load("exact_match")

    
    def make_prompt(self, resume_text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": Constants.EXTRACTION_SYSTEM_PROMPT_TEMPLATE
            },
            {
                "role": "user",
                "content": Constants.EXTRACTION_USER_PROMPT_TEMPLATE.replace("<resume_text>", resume_text)
            },
        ]
    
    def make_prompt_with_assistant(self, resume_text: str, structured_json: str) -> List[Dict[str, str]]:
        messages = self.make_prompt(resume_text)
        messages.append({
            "role": "assistant",
            "content": structured_json
        })
        return messages
    
    def generate(self, prompt: Any) -> str:
        ollama = Ollama(self.model_name)
        try:
            if isinstance(self.model_name, str):

                if(not ollama.exists()):
                    ollama.pull()

                if(isinstance(prompt, list)):
                    response = ollama.chat(prompt)
                    if response is not None:
                        return response["message"]["content"]
                    else:
                        raise Exception("No response from the model. Please check the model and try again.")
                elif(isinstance(prompt, str)):
                    response = ollama.chat(self.make_prompt(prompt))
                    if response is not None:
                        return response["message"]["content"]
                    else:
                        raise Exception("No response from the model. Please check the model and try again.")
                else:
                    raise Exception("Invalid prompt type. Please provide a string or a list of messages.")
                
            else:
                raise Exception("Model name is not a string. Please provide a valid model name.")
            
        except Exception as e:
            raise Exception(f"ResumeExtractionError: {str(e)}")
        
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict|None:
        return self.rouge_metrics.compute(predictions=predictions, references=references)
    
    def evaluate_exact_match(self, predictions: List[str], references: List[str]) -> Dict|None:
        return self.exact_match_metrics.compute(predictions=predictions, references=references)
    
    def evaluate_valid_json(self, predictions: List[str]) -> Dict:
        valid_count = 0
        invalid_json_idx = []
        for idx, prediction in enumerate(predictions):
            if ResumeUtil.valid(prediction):
                valid_count += 1
            else:
                invalid_json_idx.append(idx)
        return {
            "valid_json_count": valid_count, 
            "valid_json_score": valid_count / len(predictions) if predictions else 0, 
            "invalid_json_idx": invalid_json_idx
        }
    
    def evaluate(self, predictions: List[str], references: List[str], output_file=None) -> Dict:
        scores = {}
        if not len(predictions) or not len(references):
            scores = {"error": "predictions or references are empty"}
            if output_file:
                Utility.write_to_file(output_file, json.dumps(scores))
            return scores
        
        if len(predictions) != len(references):
            scores = {"error": f"Length of prediction {len(predictions)} does not match with the length of reference {len(references)}"}
            if output_file:
                Utility.write_to_file(output_file, json.dumps(scores))
            return scores
        
        total_test_length = len(predictions)
        rouge_scores = self.evaluate_rouge(predictions, references)
        exact_match_scores = self.evaluate_exact_match(predictions, references)
        valid_json_scores = self.evaluate_valid_json(predictions)
        
        scores = {
            "total_tests": total_test_length,
            "rouge": rouge_scores,
            "exact_match": exact_match_scores,
            "valid_json": valid_json_scores
        }

        if output_file:
            Utility.write_to_file(output_file, json.dumps(scores))
        return scores
