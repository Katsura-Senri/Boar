from .base import Algorithm
import json
import random

def get_example(self, shots=3, algorithm = 'cot_prompt'):
        self.example_root="benchmarks/data/prompts"
        if shots == 0:
            return []
        # Retrieves the example specified by name
        with open(f"{self.example_root}/gsm8k.json") as f:
            examples = json.load(f)[algorithm]

            # for zero-shot-cot, we using shots as prompt indexer
            if 'zero-shot-cot' in algorithm:
                examples = examples[shots % len(examples)]
            else:
                examples = examples.split('\n\n')
                if shots < len(examples):
                    examples = random.sample(examples, shots)
                    
        return examples
class Prompt(Algorithm):
    """
    single-step prompt:
    including direct prompt , few-shots in-context learning, chain-of-thought (CoT), zero-shot CoT, least-to-most, 
    """
    
    def __init__(self, method='standard'):
        self.PROMPT_METHODS = ['standard', 'cot', 'zero-shot-cot', 'least-to-most']
        self.method = method
        self.examples= get_example(self, shots=5, algorithm = 'standard_prompt')
        self.question_template=self.get_question_template()
        assert self.method in self.PROMPT_METHODS, f"Please choosing from {'; '.join(self.PROMPT_METHODS)}"

        self.prompt_name = method + '_prompt'
        
        print(method)

    def get_question_template(self):
        # formulate the input
            if (self.method== 'cot') :

                template = "Now,you should answer question below, you should say 'step i' before your every steps\nQ: {question}\nA:"
            else:
                template = "Q: {question}\nA:"
            return template
    def do_reasoning(self, question, context=None) -> str:
        # if we use multiple examples, we need to transform list to a str
        if isinstance(self.examples, list):
            inputs = '\n'.join(self.examples) +'\n'
        elif self.examples:
            inputs = self.examples + '\n'
        else: 
            inputs = ""
        # TODO: zero-shot-cot + standard examples
        if context:
            inputs += self.question_template.format(context=context, question=question)
        else:
            inputs += self.question_template.format(question=question)
        print(inputs)    
        responds = self.model.generate([inputs]).text[0]

        print(responds)   
        return responds
