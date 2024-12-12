import requests
import json
from .utils import Env, REACT_TEMPLATE
from .base import Algorithm

class RPMAD(Algorithm):
    def __init__(self, method):
        self.method = method
        self.base_config = json.load(open("./algorithms/MAD/config4all.json", "r"))
        self.config = self.base_config.copy()
        self.role_names = ["affirmative", "negative", "moderator", "judge"]
        self.prompt_name = 'standard_prompt'


    def set_model(self, models):
        self.affirmative = models[0]
        self.negative = models[1]
        self.moderator = models[2]
        self.judge = models[3]
        self.models = [self.affirmative,self.negative,self.moderator,self.judge]


    def do_reasoning(self, question, context, rounds=3):
        self.config = self.base_config.copy()
        if context:
            inputs = self.question_template.format(context=context, question=question)
        else:
            inputs = self.question_template.format(question=question)
        self.config['debate_topic'] = inputs
        print(f"\n===== Debate Topic =====\n")
        print(inputs)
        self.init_prompt()
        self.memory_lists = [[] for model in self.models]

        # start: set meta prompt
        self.set_meta_prompt()

        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.add_event("affirmative", self.config['affirmative_prompt'])
        self.aff_ans = self.affirmative.generate_with_messages(self.memory_lists[0]).text[0]
        self.add_memory("affirmative", self.aff_ans)
        self.config['base_answer'] = self.aff_ans

        self.add_event("negative", self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.generate_with_messages(self.memory_lists[1]).text[0]
        self.add_memory("negative", self.neg_ans)


        self.add_event("moderator",self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.generate_with_messages(self.memory_lists[2]).text[0]
        self.add_memory("moderator", self.mod_ans)
        self.mod_ans = eval(self.mod_ans)


        for round in range(rounds - 1):

            if self.mod_ans["debate_answer"] != '':
                break
            else:
                print(f"===== Debate Round-{round + 2} =====\n")
                self.add_event("affirmative", self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.affirmative.generate_with_messages(self.memory_lists[0]).text[0]
                self.add_memory("affirmative", self.aff_ans)

                self.add_event("negative",self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.negative.generate_with_messages(self.memory_lists[1]).text[0]
                self.add_memory("negative", self.neg_ans)

                self.add_event("moderator", self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round+2)))
                self.mod_ans = self.moderator.generate_with_messages(self.memory_lists[2]).text[0]
                self.add_memory("moderator", self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True

        # ultimate deadly technique.
        else:
            aff_ans = self.memory_lists[0][2]['content']
            neg_ans = self.memory_lists[1][2]['content']

            # extract answer candidates
            self.add_event("judge", self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = self.judge.generate_with_messages(self.memory_lists[3]).text[0]
            self.add_memory("judge", ans)

            # select one from the candidates
            self.add_event("judge", self.config['judge_prompt_last2'])
            ans = self.judge.generate_with_messages(self.memory_lists[3]).text[0]
            self.add_memory("judge", ans)

            ans = eval(ans)
            if ans["debate_answer"] != '':
                self.config['success'] = True
                # save file
            self.config.update(ans)

        self.print_answer()
        return self.config["debate_answer"]

    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def set_meta_prompt(self):
        self.memory_lists[0].append({"role": "system", "content": self.config['player_meta_prompt']})
        self.memory_lists[1].append({"role": "system", "content": self.config['player_meta_prompt']})
        self.memory_lists[2].append({"role": "system", "content": self.config['moderator_meta_prompt']})
        self.memory_lists[3].append({"role": "system", "content": self.config['moderator_meta_prompt']})

    def add_event(self, role_name, event):
        index = self.role_names.index(role_name)
        self.memory_lists[index].append({"role": "user", "content": f"{event}"})

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def add_memory(self, role_name, memory):
        index = self.role_names.index(role_name)
        self.memory_lists[index].append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {role_name} -----\n{memory}\n")

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])



