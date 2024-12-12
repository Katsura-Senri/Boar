import os
import time
import numpy as np
from litellm import completion
from typing import Literal, Optional, Union

from .base import GenerateOutput, LanguageModel
from .utils import get_api_key
from together import Together
import warnings

class opensource_API_models(LanguageModel):
    """
    NOTE: adding "together_ai/" in front of the model name in https://docs.together.ai/docs/inference-models before calling
    """
    def __init__(self, model:str, max_tokens:int=2048, temperature=0.0, additional_prompt=None):
        # self.model = 'together_ai/' + model
        self.model = model
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = get_api_key()
        self.additional_prompt = additional_prompt
        self.client = Together(api_key=self.api_key)

    def chat_format_prompt(self, prompt):
        return 
        
    def generate(self,
                 prompt: Optional[Union[str, list[str]]],
                 max_tokens: int = None,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 rate_limit_per_min: Optional[int] = None,
                 stop: Optional[str] = None,
                 logprobs: Optional[int] = None,
                 temperature = 1,
                 additional_prompt = None,
                 retry = 64,
                 use_chat_api = True,
                 **kwargs) -> GenerateOutput:
        
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        stop = [stop] if stop is not None and isinstance(stop, str) else stop

        # check prompt
        inputs = prompt if isinstance(prompt, list) else [prompt]
        assert num_return_sequences > 0, 'num_return_sequences must be a positive value'
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
            
        elif additional_prompt is not None and self.additional_prompt is not None:
            warnings.warn("Warning: additional_prompt set in constructor is overridden.")

        responses, res_logprobs, output_tokens = [], [], []
        for each_prompt in inputs:
            success = 0
            for i in range(1, retry + 1):
                try:
                    # sleep several seconds to avoid rate limit
                    if rate_limit_per_min is not None:
                        time.sleep(60 / rate_limit_per_min)

                    if use_chat_api:
                        res = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": each_prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            n=num_return_sequences,
                            stop=stop,
                            logprobs=1,
                        )        
                    else:
                        res = self.client.completions.create(
                            model=self.model,
                            prompt=each_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            n=num_return_sequences,
                            stop=stop,
                            logprobs=1,
                        )     
                    for i in range(num_return_sequences):
                        if use_chat_api:
                            responses.append(res.choices[i].message.content)
                        else:
                            responses.append(res.choices[i].text)
                        res_logprobs.append(res.choices[i].logprobs.token_logprobs)
                        output_tokens.append(res.choices[i].logprobs.tokens)
                    success = 1
                    break
                except Exception as e:
                    warnings.warn(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            if success == 0:
                raise RuntimeError(f"TogetherAImodel failed to generate output, even after {retry} tries")
            
        if logprobs == True:
            output = GenerateOutput(
                text=responses,
                log_prob=res_logprobs,
                str_each_token = output_tokens
            )
        else:
            output = GenerateOutput(
                text=responses
            )
        return output

    def generate_with_messages(self,
                               messages,
                               max_tokens: int = None,
                               top_p: float = 1.0,
                               num_return_sequences: int = 1,
                               rate_limit_per_min: Optional[int] = 60,
                               stop: Optional[str] = None,
                               logprobs: Optional[int] = None,
                               temperature=None,
                               additional_prompt=None,
                               retry=64,
                               **kwargs) -> GenerateOutput:

        # check hyper-parameters
        gpt_temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        logprobs = 0 if logprobs is None else logprobs
        stop = [stop] if stop is not None and isinstance(stop, str) else stop
        print("API Key:", self.api_key)
        print("Model:", self.model)
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            warnings.warn("Warning: additional_prompt set in constructor is overridden.")

        responses = []

        success = 0
        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                response = completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=gpt_temperature,
                    top_p=top_p,
                    stop=stop
                )

                for i in range(num_return_sequences):
                    responses.append(response.choices[i].message.content)
                success = 1
                break
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)
        if success == 0:
            # after 64 tries, still no luck
            raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
        return GenerateOutput(text=responses)
    
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> list[np.ndarray]:
        acc_probs = []
        for content in contents:
            acc_probs.append(0)
            full_prompt = prefix + content
            res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
            cumulative_text = ""
            tokens = res_echo.prompt[0].logprobs.tokens
            for i in range(len(tokens)):
                token = tokens[i]
                cumulative_text += token
                if len(cumulative_text) > len(prefix):
                    logprobs = res_echo.prompt[0].logprobs.token_logprobs[i]
                    acc_probs[-1] += logprobs
        return acc_probs

    def get_next_token_logits(self, 
                              prompt: str | list[str], 
                              candidates: list[str] | list[list[str]], 
                              postprocess: str | None = None, 
                              **kwargs) -> list[np.ndarray]:
        # assert postprocess in ['logits', 'logprobs', 'log_softmax', 'probs', 'softmax']
        prompts = [prompt] if isinstance(prompt, str) else prompt
        logprobs = []
        if isinstance(candidates[0], str): 
            candidates = [candidates] * len(prompt)
        for prompt, candidate_list in zip(prompts, candidates):
            logprob_list = []
            for candidate in candidate_list:
                full_prompt = prompt + candidate
                res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
                tokens = res_echo.prompt[0].logprobs.tokens
                last_token = tokens[-1]
                if candidate in last_token:
                    logprob_list.append(res_echo.prompt[0].logprobs.token_logprobs[-1])
                else:
                    matched_tokens = []
                    for i in range(-1, -len(tokens), -1):
                        matched_tokens.insert(0, tokens[i])
                        if candidate in ''.join(matched_tokens):
                            logprob_list.append(res_echo.prompt[0].logprobs.token_logprobs[i])
                            warnings.warn("warning: candidate {} has more than one token with {} tokens".format(candidate, -i))
                            break
            logprobs.append(logprob_list)
        return logprobs
    
# the following is used to test this script
if __name__ == '__main__':
    os.environ["TOGETHERAI_API_KEY"] = ""
    model = opensource_API_models(model="meta-llama/Meta-Llama-3.1-405B", max_tokens=100)
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?'], temperature= 0.7, top_p=0.7, top_k=50))
    print(model.get_loglikelihood("How can I goto Shanghai from Beijing?", [" By bicycle.", " By bus.", " By train.", " By air.", " By rocket.", " By spaceship."]))
    print(model.get_next_token_logits(["The capital of UK is", "The capital of France is", "The capital of Russia is"], ["Paris", "London", "Moscow"]))