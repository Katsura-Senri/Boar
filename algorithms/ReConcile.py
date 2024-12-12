from .base import Algorithm
from .utils import Env, RECONCILE_TEMPLATE, parse_json
from collections import Counter
# Input output 接口
class ReConcile(Algorithm):
    def __init__(self, method='ReConcile', output_type='weighted'):
        self.method = method
        self.prompt_name = method + '_prompt'
        self.env = Env()
        self.models = None
        self.model_name_dict = {}
        self.convincing_samples = {}
        self.dataset = None
        self.datasets = [i.lower() for i in ["SQA", "GSM8k", "ECQA", "Aqua"]]
        self.output_type = output_type # majority_vote, weighted_max
        
    def set_dataset(self, dataset):
        try:
            dataset = dataset.lower()
            
            if dataset not in self.datasets:
                raise ValueError(f"Dataset {dataset} is not supported.")
            else:
                self.dataset = dataset
        except:
            raise ValueError(f"Dataset {dataset} is not supported.")
        
    def set_model(self, model):
        pass
    
    def set_models(self, models: list, convincing_samples={}):
        """
        Set multiple models to be used for reasoning.
        :param models: List of models where each model has a generate method.
        :param convincing_samples: Convincing samples to be used in the reasoning process.
            convincing_samples Example:
            {
                "model1": [{
                    "train_sample": {
                        "index": 623,
                        "question": "Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?",
                        "answer": "15",
                        "gold_explanation": "Each drink uses 1.5 cups because 1/4 cup + 1 and 1/4 cup = 1.5 cups. The pitcher contains 12 total drinks because 18 / 1.5 = 12. There are 15 cups of lemonade in the pitcher because 12 x 1 and 1/4 = 15."
                    },
                    "prediction_pre": {
                        "question_understanding": "Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?",
                        "answer_reasoning": "To answer this question, we first need to find the total amount of lemonade in 18 drinks. We know that each drink uses 1 and 1/4 cups of lemonade, so 18 drinks would use 18 * 1 and 1/4 = 25 and 1/2 cups of lemonade. Since each drink also uses 1/4 cup of iced tea, the total amount of iced tea in 18 drinks would be 18 * 1/4 = 4.5 cups. Therefore, the total amount of lemonade and iced tea in the pitcher is 25 and 1/2 + 4.5 = 29 and 1/2 cups. Since we know that there are 18 cups of this drink in the pitcher, the number of cups of lemonade must be 29 and 1/2 / 18 = 1 and 1/2 cups.",
                        "answer": "1 and 1/2",
                        "confidence_level": "0.9",
                        "confidence_explanations": "I am confident in my answer because I followed the steps logically and used the information provided in the question to arrive at my answer. I also checked my answer by making sure that it made sense in the context of the question."
                    },
                    "prediction_post": {
                        "question_understanding": "Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?",
                        "answer_reasoning": "Each drink uses 1.5 cups because 1/4 cup + 1 and 1/4 cup = 1.5 cups. The pitcher contains 12 total drinks because 18 / 1.5 = 12. There are 15 cups of lemonade in the pitcher because 12 x 1 and 1/4 = 15.",
                        "answer": "15",
                        "confidence_level": 1,
                        "confidence_explanations": "I am 100% confident in my answer because I followed the instructions carefully and used the correct mathematical equations."
                    }
                }, ...],
                "model2": [...],
                "model3": [...]
            }
        
        """
        self.models = models
        for model in models:
            self.model_name_dict[model.model] = model
            
        # TODO where to set convincing_samples
        if convincing_samples:
            self.convincing_samples = convincing_samples
         
    def set_example(self, examples):
        """
        Set the examples to be used in the reasoning process.
        """
        self.examples = examples
        
    def set_question_template(self, question_template):
        """
        Set the question template for reasoning.
        """
        self.question_template = question_template
        
    def set_convincing_samples(self, convincing_samples):
        """
        Set the convincing samples to be used in the reasoning process.
        """
        self.convincing_samples = convincing_samples
        
    # def init_chat_history(self):
    #     chat_history = []
        
        

    def do_reasoning(self, question, context=None, options=None, rounds=3) -> str:
        """
        Phase1: Initial Response Generation
        For each model, we generate an initial response based on the provided question and examples.
        """
        if not self.models:
            raise ValueError("No models are set. Please set models using set_models().")
        
        
        chat_history = {}
        
        # 1. Generate initial responses for each model        
        init_responses = self.get_initial_response(question, options)
        vote_info, top1_answer, weighted_vote, weighted_max = self.vote(init_responses)
        chat_history_round_0 = chat_history.get('round_0', {})
        chat_history_round_0['responses'] = init_responses
        chat_history_round_0['vote_info'] = vote_info
        chat_history_round_0['weighted_vote'] = weighted_vote
        chat_history_round_0['weighted_max'] = weighted_max
        chat_history_round_0['majority_vote'] = top1_answer
        chat_history['round_0'] = chat_history_round_0
        
        # 2. Multi-Round Discussion
        for r in range(1, rounds+1):
            debate_responses = self.multi_model_debate(question, chat_history, r, options=options)
            vote_info, top1_answer, weighted_vote, weighted_max = self.vote(debate_responses)
            chat_history_round_r = chat_history.get(f'round_{r}', {})
            chat_history_round_r['responses'] = debate_responses
            chat_history_round_r['vote_info'] = vote_info
            chat_history_round_r['weighted_vote'] = weighted_vote
            chat_history_round_r['weighted_max'] = weighted_max
            chat_history_round_r['majority_vote'] = top1_answer
            chat_history[f'round_{r}'] = chat_history_round_r

            
        # 3. answer
        """
        {
            'model1': '1',
            'model2': '1',
            ...
            'weighted_vote': '1',
            'majority_vote': '1'
        }
        """
        final_round_responses = chat_history[f'round_{rounds}']['responses']
        final_round_chat_history = chat_history[f'round_{rounds}']
        model_answers = {k: v['answer'] for k,v in final_round_responses.items() if k != 'vote_info'}
        answer_dict = {
            **model_answers,
            'weighted_max':final_round_chat_history['weighted_max'],
            'majority_vote':final_round_chat_history['majority_vote']
        }
        
        try:
            return answer_dict[self.output_type]
        except:
            raise NotImplementedError(f"Output type {self.output_type} is not supported.")
        
        

    def get_initial_response(self, question, options=None) -> dict:
        """
        Generate the initial response for each model.
        {
            "model1": {
                "model": "model1",
                "response": "response1",
                "parsed_response": {"answer": "1", "reasoning": "reasoning1"}
                "answer": "1",
                "confidence_level": 0.9
            },
            ...
        }
        """
        responses = {}
        for model in self.models:
            responses[model.model] = {}
            response = model.generate(
                [self.prepare_context(question, options)], 
                max_length=1024,
            ).text[0]
            parsed_json = parse_json(response)
            model_response_info = {
                "model": model.model,
                "question": question,
                "response": response,
                "options": options,
                "answer": str(parsed_json.get('answer', '')),
                "confidence_level": str(parsed_json.get('confidence_level', '0')),
                "explanation": str(parsed_json.get('reasoning', ''))
            }
            responses[model.model] = model_response_info
        return responses
        
        
    def prepare_context(self, question, options=None):
        """
        Prepare the context for the reasoning process.
        Param intervene is removed (was implemented in original code)
        """
        dataset = self.dataset.lower()
        assert dataset in self.datasets
        context = []
        
        if self.convincing_samples:
            for cs in self.convincing_samples:
                context.append(f"Q: {cs['train_sample']['question']}\nA:" + str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']}))
        else:
            context.append("Q: " + question)
            
        if options is not None:
            context.append("The options are: {}. Please select an option as your answer.".format(" ".join(test_sample["options"])))
       
        context.append("Please answer the question with step-by-step reasoning.")
        context.append("\nAlso, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right.")
        context.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
        context.append(self.get_instruction_by_current_dataset())
        context.append("Do not output irrelevant content.")
        context_str = '\n'.join(context)
        return context_str
    
    def prepare_debate_instruction(self, round, chat_history):
        # if round is 1, no debate_prompt like vote result of previous round
        # if round == 1:
        #     debate_prompt = ''
        # else:
        debate_prompt = self.get_vote_info_insrtuction(chat_history[f'round_{round-1}']['vote_info'])
            
        additional_instruction = [
            "\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.",
            "Clearly states that which pointview do you agree or disagree and why.\n\n",
            debate_prompt,
            "Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format."
        ]
        
        return additional_instruction
        
    def get_vote_info_insrtuction(self, vote_info):
        
        instructions = []
        for vote in list(vote_info.values()):
            vote_number = vote['vote_number']
            answer = vote['answer']
            explanations = "\n".join(e['explanation'] for e in vote['explanations'])
            instruction = f"""There are {vote_number} agents think the answer is {answer}.
                {explanations} 
            """
            instructions.append(instruction)
            
        return ('\n\n').join(instructions)
            
        
    
    def multi_model_debate(self, question, chat_history, round, options=None):
        """
        Model debate process.
        Multiple models debate
        """
        responses = {}
        for model in self.models:
            responses[model.model] = {}
            response = model.generate(
                [self.prepare_context(question, options) + 
                 ' '.join(self.prepare_debate_instruction(round, chat_history))], 
                max_length=1024,
            ).text[0]
            parsed_json = parse_json(response)
            model_response_info = {
                "model": model.model,
                "question": question,
                "response": response,
                "options": options,
                "answer": str(parsed_json.get('answer', '')),
                "confidence_level": str(parsed_json.get('confidence_level', '0')),
                "explanation": str(parsed_json.get('reasoning', ''))
            }
            responses[model.model] = model_response_info
        return responses
            

    
    def vote(self, responses):
        # predictions, explanations, confidence_levels, rounds
        
        """
        dict
        {
            '1': {
                'vote_number': 2,
                'explanations': [
                    {'model': 'model1', 'answer': '1', 'explanation': 'explanation1'},
                    {'model': 'model2', 'answer': '1', 'explanation': 'explanation2'},
                ],
                'answer': '1',
            },
        }
        """
        vote_info_all = {}
        
        predictions = list(set([v['answer'] for k,v in responses.items()]))
        weighted_vote = {}
        
        for res in list(responses.values()):
            ans = res['answer']

            vote = vote_info_all.get(ans, {})
            vote['vote_number'] = vote.get('vote_number', 0) + 1
            explanations = vote.get('explanations', [])
            explanations.append({
                'model': res['model'],
                'answer': res['answer'],
                'explanation': res['explanation']
            })
            vote['explanations'] = explanations
            vote['answer'] = ans
            vote_info_all[ans] = vote
            
            confidence = self.trans_confidence(res['confidence_level'])
            if ans not in weighted_vote:
                weighted_vote[ans] = confidence + 1e-5  # 增加一个小值以防止精度丢失
            else:
                weighted_vote[ans] += confidence
        
        weighted_max = max(weighted_vote, key=weighted_vote.get)
        vote_counter = Counter(predictions)
        
        # TODO top2 ?
        top2_vote = vote_counter.most_common(2)
        top1_answer = top2_vote[0][0]
        # construct vote_info for debate_instruction if multiple vote results
        vote_info = {}
        if len(top2_vote) > 1:
            for v in top2_vote:
                vote_info[v[0]] = vote_info_all[v[0]]
        else:
            vote_info[top1_answer] = vote_info_all[top1_answer]
        
        return vote_info, top1_answer, weighted_vote, weighted_max
                
    
    @staticmethod
    def trans_confidence(x):
        x = float(x)
        if x <= 0.6: return 0.1
        if 0.8 > x > 0.6: return 0.3
        if 0.9 > x >= 0.8: return 0.5
        if 1 > x >= 0.9: return 0.8
        if x == 1: return 1
        
        
        
    def get_instruction_by_current_dataset(self):
        dataset = self.dataset.lower()
        if dataset == "sqa":
            instruction = "Only answer yes or no in the \"answer\" field."
        elif dataset == "gsm8k":
            instruction = "Only place a single numeric value in the \"answer\" field."
        elif dataset == "ecqa":
            instruction = "Only place 1,2,3,4,5 representing your choice in the \"answer\" field."
        elif dataset == "aqua":
            instruction = "Only place A,B,C,D,E representing your choice in the \"answer\" field."
        else:
            raise NotImplementedError(f"Dataset {dataset} is not supported.")
            
        return instruction
