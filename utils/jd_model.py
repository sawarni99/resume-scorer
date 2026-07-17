
from utils.utility import Utility
from utils.constants import Constants
from typing import List, Dict, Any
from utils.ollama_util import Ollama
from utils.jd_util import JobDescriptionUtil
from datatypes.jd_data_type import JobDescription
import evaluate
import json

class JobDescriptionModel:
    model_name = None
    utility = Utility()

    def __init__(self, model_name):
        self.model_name = str(model_name)
        self.rouge_metrics = evaluate.load("rouge")
        self.exact_match_metrics = evaluate.load("exact_match")

    
    def make_prompt(self, jd_text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": Constants.JD_EXTRACTION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": Constants.JD_EXTRACTION_USER_PROMPT_TEMPLATE.replace("<jd_text>", jd_text)
            },
        ]
    
    def make_prompt_with_assistant(self, jd_text: str, structured_json: str) -> List[Dict[str, str]]:
        messages = self.make_prompt(jd_text)
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
                    print(f"Model is not present in ollama. Pulling the model:\n{self.model_name}")
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
            raise Exception(f"JobDescriptionExtractionError: {str(e)}")
        
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict|None:
        return self.rouge_metrics.compute(predictions=predictions, references=references)
    
    def evaluate_exact_match(self, predictions: List[str], references: List[str]) -> Dict|None:
        return self.exact_match_metrics.compute(predictions=predictions, references=references)
    
    def evaluate_valid_json(self, predictions: List[str]) -> Dict:
        valid_count = 0
        invalid_json_idx = []
        for idx, prediction in enumerate(predictions):
            if JobDescriptionUtil.valid(prediction):
                valid_count += 1
            else:
                invalid_json_idx.append(idx)
        return {
            "valid_json_count": valid_count, 
            "valid_json_score": valid_count / len(predictions) if predictions else 0, 
            "invalid_json_idx": invalid_json_idx
        }
    
    def evaluate_valid_exp_yrs(self, predictions: List[str]) -> Dict:
        non_numeric_max_exp = []
        non_numeric_min_exp = []

        jd_predictions = [JobDescription.model_validate_json(prediction) for prediction in predictions]

        for idx, jd in enumerate(jd_predictions):
            is_min_exp_number = JobDescriptionUtil.min_exp_is_number_or_none(jd)
            is_max_exp_number = JobDescriptionUtil.max_exp_is_number_or_none(jd)
            if not is_min_exp_number:
                non_numeric_min_exp.append(idx)
            if not is_max_exp_number:
                non_numeric_max_exp.append(idx)

        return {
            "number_non_numeric_min_exp": len(non_numeric_min_exp),
            "non_numeric_min_exp_idx": non_numeric_min_exp,
            "number_non_numeric_max_exp": len(non_numeric_max_exp),
            "non_numeric_max_exp_idx": non_numeric_max_exp,

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
        valid_exp_yrs = self.evaluate_valid_exp_yrs(predictions)
        
        scores = {
            "total_tests": total_test_length,
            "rouge": rouge_scores,
            "exact_match": exact_match_scores,
            "valid_json": valid_json_scores,
            "valid_exp_yrs": valid_exp_yrs
        }

        if output_file:
            Utility.write_to_file(output_file, json.dumps(scores))
        return scores
