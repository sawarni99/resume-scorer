import fitz
from pydantic import BaseModel
import re

class FileUtil(BaseModel):
    
    file_text: str

    def __remove_special_chars(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s/,.:;'\"~!@#$%^&*()\[\]{}\\\|_+\-]", "", text)

    def extract_text(self, file_path: str) -> str:

        # Extrating data...
        doc = fitz.open(file_path)
        result:str = ""

        # Joining all the extracted pages in one text...
        for page in doc:
            text = "".join(page.get_text("text"))
            result += text

        # Sanitizing the data...
        self.file_text:str = self.__remove_special_chars(result)
        
        return self.file_text


