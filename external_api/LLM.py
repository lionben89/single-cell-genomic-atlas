from abc import ABC, abstractmethod
import json
from huggingface_hub import InferenceClient
from huggingface_hub import login as huggingface_login

from openai import OpenAI

with open('./external_api/keys.json',"rb") as json_data:
    keys = json.load(json_data)
    json_data.close()

class LLM(ABC):
    def __init__(self,**kwargs):
        super().__init__()
        self.client = self.get_client()
    
    @abstractmethod
    def get_client(self):
        pass

    @abstractmethod
    def ask(self,prompt):
        pass
    
    def make_prompt_from_df(self, df, df_context, question, max_rows=None):
        """Convert a DataFrame into a markdown-formatted table and append a question."""
        if max_rows is not None:
            if len(df) > max_rows:
                df = df.head(max_rows)
        table = df.to_markdown(index=False)
        prompt = f"""
            {df_context}

            {table}

            {question}
        """
        return prompt
    
    
class Mistral(LLM):
    def __init__(self,max_new_tokens,temperature):
        ##Imports
        super().__init__()
        huggingface_login(token=keys["hf"])
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def get_client(self):
        # LLM client
        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")  # Free chat model
        return client

    def ask(self,prompt):
        turncut_prompt = prompt[:32768-self.max_new_tokens]
        response = self.client.text_generation(turncut_prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        return response

class OpenAILLM(LLM):
    
    def __init__(self):
        ##Imports
        super().__init__()

    def get_client(self):
        # LLM client
        client = OpenAI(api_key=keys["openai"])
        return client

    def ask(self,prompt):
        response = self.client.responses.create(
            model="gpt-4o",
            input=prompt
        )

        return response.output_text


def get_llm(type,**kwargs):
    LLMs = {
        'mistral':Mistral,
        'openai':OpenAILLM,
    }
    
    llm_instance = LLMs[type](**kwargs)
    return llm_instance


