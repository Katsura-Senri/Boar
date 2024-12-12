import time
import gym
import requests
from bs4 import BeautifulSoup
import re
import ast



REACT_TEMPLATE='''Here's some examples: {examples}
You must carefully follow these rules:
1. Your answer must begin with 'Thought', 'Action', and 'Observation', and you need to number your answers sequentially.
2. Every time you give an answer, you should give one Thought, one Action, and one Observation, without anything else, but must, contain all of them! Don't just give me one Thought or one Action!
3. Each Thought, Action, and Observation must be numbered consistently with the last number I provided, and must not repeat the numbering from the earlier prompts.
4. You must only provide one step at a time.
5. Your calculation after the Action must be enclosed in square brackets [] and should not include units.
6. Don't provide cross answers. For example, don't start with 'Action' where you should give a 'Thought'. You must start with 'Thought'.
7. Don't provide any explanations or apologies, just follow the format strictly.
8. Don't using symbols like '?' in calculating.
Your task is to solve this problem, think carefully: {question}'''

RECONCILE_TEMPLATE='''{question}'''

SCORE_TEMPLATE = (
    "Your task is to solve this problem: '{question}'\n"
    "We have already found these steps to solve the problem:\n"
    "{steps_now}\n"
    "And we have these possible methods for the next step:\n"
    "{methods}\n"
    "Please score each method from 1 to 10, remember, 10 means the best method (like provide a clear answer) and 1 means the worst method (like one method which repeat the previous steps), you can compare those methods and then score them.\n"
    "Remember, you must carefully follow these rules:\n"
    "Your response should be in the format 'score1 score2 ... reason1 reason2'(Remember: Numbers depends on how many methods do we have! If we have only one method, you should only return one digit!).\n"
    "Please ensure that the number of scores matches the number of methods!!!\n"
    "Please provide distinct scores.\n"
    "The scores MUST appear on the first line, don't say anything before them!\n"
    "Please provide detailed reasons after all scores, explaining why you gave each score, including where each points were earned and where they were deducted, why methods with higher scores are better than those with lower scores, and DO NOT include any additional content.\n"
)

ONE_STEP_CHECK_TEMPLATE = (
    "Your task is to solve this problem: '{question}'\n"
    "We have already found these steps to solve the problem:\n"
    "{steps_now}\n"
    "Check the last step carefully!\n"
    "1. If you think the last step has already solved the problem, please validate the answer for the question; if the answer is right, return 'The answer is xxx' (xxx is the answer, a number).\n"
    "2. If you think the last step is still an intermediate step, has not solved the whole problem yet, but the step is meaningful or promising to solve the problem, return 'Let's continue!', and if you think that the step is meaningless, return 'Let's stop and rethink...'.\n"
    "3. If you think you can simplify the calculation and get the final result of the question, return 'The answer is xxx' (xxx is the answer, a number)!\n"
    "Please follow these formats strictly, return the content within the quotation marks before, and provide your reason in the next line.\n"
)

TOT_REASONING_TEMPLATE = (
    "Here's some examples: \n"
    "{examples}\n"
    "Your task is to solve this problem:\n"
    "{question}\n"
    "Remember, you MUST carefully follow these rules:\n"
    "1. You should follow the examples strictly; only give one step which is different from the previous steps, don't solve the problem in Step 1!\n"
    "2. If you can, you could give 2-3 different methods, just one step, and must be different from each other!\n"
    "3. Please follow the format of the examples strictly (your answer should follow a 'Method i:' format closely, and do not include 'Step').\n"
    "4. If there is a '?' in the previous step, you MUST solve it and calculate it.\n"
    "5. Your answer MUST not be the same as the previous steps.\n"
    "6. Your Methods MUST be different from each other and the previous steps.\n"
)

TWO_STEP_CHOOSE_TEMPLATE = (
    "Your task is to solve this problem: '{question}'\n"
    "We have already found these steps to solve the problem:\n"
    "{steps_now}\n"
    "1. (You should not answer this if we are still at early steps like Step 1 - Step 3) If you think the last step has already solved the problem, return '1'\n"
    "2. (You should not answer this if we are still at late steps like Step {max_step_half} - Step {max_step}) If you think the last step is still an intermediate step, has not solved the whole problem yet, return '2'\n"
    "Please follow these formats strictly, return the content without the quotation marks, and provide your reason in the next line.\n"
)

"""
If you want to see more backtracking and make it more like an original Tree of Thought, consider splitting the two points into separate ones, like what I commented out.
If you want to speed up reasoning or achieve better performance, you might prefer to keep the current approach.
"""

FIRST_CASE_TEMPLATE = (
    "Some people were asked to solve this problem: '{question}'\n"
    "Firstly, one person got this result:\n"
    "{steps_now}\n"
    "And another person got this result:\n"
    "{response1}\n"
    "You don't need to solve the problem, you don't need to consider the steps to solve the problem, just focus on whether the final answer is correct and using the answers you got to compare:\n"
    "1. If the two final answers (two numbers) are the same, or you prefer the second answer, return 'The answer is xxx' (xxx is the answer, a number).\n"
    "2. If the two final answers (two numbers) are different, if you think the first answer is correct, return 'The answer is xxx' (xxx is the answer, a number), if you think the second answer is correct, do not return the answer, return 'Let's rethink!'.\n"
    # "3. If the two final answers (two numbers) are different, if you think the second answer is correct, do not return the answer, return 'Let's rethink!'.\n"
    "Please follow these formats strictly, return the content without the quotation marks, and provide your reason in the next line.\n"
)

SECOND_CASE_TEMPLATE = (
    "Your task is to solve this problem: '{question}'\n"
    "We have already found some intermediate steps, but we have not solved the whole problem yet:\n"
    "{steps_now}\n"
    "1. If you think that the steps, especially the last step, are meaningful or promising to solve the problem, return 'Let's continue!'\n"
    "2. If you think that the steps, especially the last step, are meaningless, return 'Let's stop and rethink...'.\n"
    "Please follow these formats strictly, return the content without the quotation marks, and provide your reason in the next line.\n"
)

def parse_json(model_output):
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    try:
        model_output = model_output.replace("\n", " ")
        model_output = re.search('({.+})', model_output).group(0)
        model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        result = ast.literal_eval(model_output)
    except (SyntaxError, NameError, AttributeError):
        # return "ERR_SYNTAX"
        return {}
    
    return result

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def replace_percentages(text):
    pattern = r'(\d+)%'
    
    def replace_match(match):
        number = int(match.group(1))
        return str(number / 100)

    result = re.sub(pattern, replace_match, text)
    return result

class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
    
        return isinstance(x, str)

# ReAct environment
class Env(gym.Env):

    def __init__(self):
      
        super().__init__()
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0
      
    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
      
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
      
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
      
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def search_step(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        old_time = time.time()
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
    
    def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()
        
        if "finish" in action.lower() and "]" in action.lower():
            start = action.lower().index("[") + len("[")
            end = action.lower().index("]")
            answer = action[start:end].replace(",", "").replace("?", "0").replace("x", "*")
            if '%' in answer:
                answer = replace_percentages(answer)
            try:
                self.answer = eval(answer)
                done = True
                self.obs = f"The answer is {answer}\n"
            except Exception:
                self.obs = f"Ohh, maybe something wrong...\n"

        elif "calculate" in action.lower() and "]" in action.lower():
            start = action.lower().index("[") + len("[")
            end = action.lower().index("]")
            expression = action[start:end].replace(",", "").replace("x", "*")
            if not any(char.isdigit() for char in expression):
                self.obs = f"Ohh, there is no numbers in {expression}, I can only calculate numbers...\n"
            elif expression == ' ':
                self.obs = f"Ohh, there is nothing in {expression}\n"
            elif '?' in expression:
                self.obs = f"Ohh, there is a '?' in {expression}, I can not calculate '?', I should use numbers.\n"
            elif '_' in expression:
                self.obs = f"Ohh, there is a '_' in {expression}, I can not calculate '_', I should use numbers.\n"
            else:
                if '%' in expression:
                    expression = replace_percentages(expression)
                try:
                    result = eval(expression)
                    result = round(result, 5)
                    self.obs = f"The result is {result}\n"
                except Exception as e:
                    self.obs = f"Ohh, maybe something wrong in {expression}\n"

        elif "search[" in action.lower() and action.endswith("]"):
            start = action.lower().index("search[") + len("search[")
            entity = action[start:-1]
            self.search_step(entity)

        elif "lookup[" in action.lower() and action.endswith("]"):
            start = action.lower().index("lookup[") + len("lookup[")
            keyword = action[start:-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1

        else:
            self.obs = action

        self.steps += 1

        return self.obs, reward, done, self._get_info()

    
    def get_time_info(self):
      speed = self.search_time / self.num_searches if self.num_searches else 0
      return {
          "call_speed": speed,
          "call_time": self.search_time,
          "num_calls": self.num_searches,
      }
