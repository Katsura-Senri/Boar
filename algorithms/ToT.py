import re
import json
import random
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from plotly.io import to_html
import io
from io import BytesIO
import base64
from .base import Algorithm
import textwrap
from .utils import (
    SCORE_TEMPLATE, 
    ONE_STEP_CHECK_TEMPLATE, 
    TOT_REASONING_TEMPLATE, 
    TWO_STEP_CHOOSE_TEMPLATE, 
    FIRST_CASE_TEMPLATE, 
    SECOND_CASE_TEMPLATE
)
from fileCount import setFileCount, getFileCount

def get_question_template(self):
        # formulate the input
        template = "Q: {question}\nA:"
        return template
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
class Node:
    def __init__(self, step=None, parent=None, method='tot'):
        self.method = method
        self.prompt_name = method + '_prompt'
        self.step = step
        self.children = []
        self.parent = parent
        

    def add_child(self, node):
        self.children.append(node)

    def remove_child(self, node):
        if node in self.children:
            self.children.remove(node)

    def is_leaf(self):
        return len(self.children) == 0

    def demo(self):
        return f"Node(step={self.step})"

class Tree:
    def __init__(self):
        self.root = Node("root")

    def add_steps(self, parent, steps):
        nodes = [Node(step, parent) for step in steps]
        parent.children.extend(nodes)
        return nodes

    def get_leaf_nodes(self):
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves
    
    def remove_node(self, node):
        if node == self.root:
            raise ValueError("Cannot remove the root node")
        
        for child in node.children:
            self.remove_node(child)
        
        if node.parent:
            node.parent.remove_child(node)

    def print_all_nodes(self):
        def dfs(node):
            print(node.demo())
            for child in node.children:
                dfs(child)
        
        dfs(self.root)

    def draw_tree(self):
        G = nx.DiGraph()
        
        # 构建树结构
        def add_edges(node):
            label_text = "\n".join(textwrap.wrap(node.demo(), width=20))
            G.add_node(node.step, label=label_text)
            for child in node.children:
                G.add_edge(node.step, child.step)
                add_edges(child)

        add_edges(self.root)

        for node in G.nodes:
            depth = len(nx.ancestors(G, node))
            G.nodes[node]['subset'] = depth

        # 使用 multipartite_layout 自动计算节点位置
        pos = nx.multipartite_layout(G, subset_key="subset")

        # 调整位置以实现纵向排列
        pos = {node: (pos[node][1], -pos[node][0]) for node in G.nodes}
        
        # 收集节点和边的布局信息
        x_nodes = [pos[node][0] for node in G.nodes]
        y_nodes = [pos[node][1] for node in G.nodes]
        edge_x = []
        edge_y = []

        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # 绘制边
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        # 绘制节点，节点上的标签悬停显示
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black')),
            hoverinfo='text',
            text=[G.nodes[node]['label'].replace("\n", "<br>") for node in G.nodes]
        )

        # 创建图表布局
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Tree Structure with Hover",
                            showlegend=False,
                            hovermode='closest',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        # 生成 HTML 文件
        setFileCount()
        fig.write_html(f"my-vue-app/public/test{getFileCount()}.html")
        
        # 获取 HTML 字符串
        html_str = to_html(fig, full_html=True, include_plotlyjs=False)
        html_f = '<!DOCTYPE html>' + html_str
        
        return html_f
# 示例使用
# tree = Tree()
# step1 = tree.add_steps(tree.root, ["Step 1"])[0]
# step2 = tree.add_steps(step1, ["Step 1.1", "Step 1.2"])
# step3 = tree.add_steps(tree.root, ["Step 2"])

# 绘制树形图
# tree.draw_tree()

class ToT(Algorithm):
    def __init__(self, method='tot', max_step=10):
        super().__init__()
        self.prompt_name = method + '_prompt'
        self.max_step = max_step
        self.examples= get_example(self, shots=5, algorithm = 'tot_prompt')
        self.question_template=get_question_template(self)

    def llm(self, prompt, stop=None):
        response = self.model.generate([prompt], stop=stop).text[0]
        return response
    
    def score_methods(self, question, steps_now, methods):
        """
        Score the methods, the performance is awful now...
        """
        prompt = SCORE_TEMPLATE.format(
            question=question,
            steps_now=steps_now,
            methods='\n'.join([f"Method {j + 1}: {method}" for j, method in enumerate(methods)])
        )

        response = self.llm(prompt, stop='assistant')

        response_lines = response.split('\n')

        """
        Find the first line that starts with a number.
        Make code more robust...
        """
        first_valid_line_index = None
        for i, line in enumerate(response_lines):
            if line and line[0].isdigit():
                first_valid_line_index = i
                break

        valid_line = response_lines[first_valid_line_index]

        scores = {}
        valid_line = valid_line.split()
        valid_line = valid_line[:len(methods)]
        for i, line in enumerate(valid_line):
            score = int(float(line.strip()) )
            scores[i] = score

        sorted_methods = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        return sorted_methods
    
    def one_step_check_answer(self, question, steps_now):
        """
        Check if the answer is correct, promising or incorrect.
        """
        prompt = ONE_STEP_CHECK_TEMPLATE.format(
            question=question,
            steps_now=steps_now
        )

        response = self.llm(prompt, stop='assistant')

        if "simplify the calculation" in response or "simplify the problem" in response or "double-" in response or "re-examine" in response:
            """
            A trick...
            Experiments show that when this phrase is mentioned, an answer has already been obtained.
            But the LLM seems a bit hesitant, so returning this would be correct...
            """
            return "The answer is..."
        
        if "we haven't considered" in response or "we have not considered" in response:
            """
            A trick...
            Experiments show that when this phrase is mentioned, The answer has been wrong.
            But the LLM seems a bit hesitant, so returning this would be correct...
            """
            return "Let's stop and rethink..."
        
        """
        A trick...
        Make code more robust...
        """
        keywords = ["The answer is", "rethink", "Let's continue!"]

        res = response.split('\n')[0]
    
        for keyword in keywords:
            if keyword in res:
                return res

        return "Let's continue!"

    def two_step_check_answer(self, question, steps_now):
        """
        Firstly, check if the problem has been solved.
        Secondly, check if the steps now are correct.
        This method maybe more robust than one_step_check_answer, however, it costs more.
        """
        max_step_half = int(self.max_step / 2)

        prompt = TWO_STEP_CHOOSE_TEMPLATE.format(
            question=question,
            steps_now=steps_now,
            max_step=self.max_step,
            max_step_half=max_step_half
        )

        response = self.llm(prompt, stop='assistant')

        if '1' in response.split('\n')[0]:
            prompt1 = (f"Your task is to solve this problem: '{question}'" + '\n' +
                  "Please solve the problem step by step!" + '\n'
                )
            response1 = self.llm(prompt1, stop='assistant')

            prompt = FIRST_CASE_TEMPLATE.format(
                question=question,
                response1=response1,
                steps_now=steps_now
            )
            
            response = self.llm(prompt, stop='assistant')

            if "Let's rethink!" in response or "he second answer is correct" in response:
                response = "Let's rethink!"

            response = response

        elif '2' in response.split('\n')[0]:
            prompt = SECOND_CASE_TEMPLATE.format(
                question=question,
                steps_now=steps_now
            )
            response = self.llm(prompt, stop='assistant')

        else:
            return self.one_step_check_answer(question, steps_now)

        keywords = ["The answer is", "rethink", "Let's continue!"]

        res = response.split('\n')[0]
    
        for keyword in keywords:
            if keyword in res:
                return res

        return "Let's continue!"
    
    def extract_methods(self, responses, i):
        methods = re.findall(r'Method \d+:([\s\S]*?)(?=Method \d+:|\Z)', responses)
        
        cleaned_methods = [method.strip().replace('\n', ' ').replace(f'Step {i + 1}:', ' ').strip() for method in methods]

        return cleaned_methods
    
    def is_valid_ranking(self, ranks, num):
        """
        Check if the ranks list is a valid permutation of numbers from 1 to num.
        Make code more robust...
        """
        try:
            ranks = list(ranks)
        except ValueError:
            return False

        return sorted(ranks) == list(range(0, num))
    
    def construct_prompt(self, question, current_node, step):
        """
        The prompt for one step reasoning.
        """
        prompted_input = TOT_REASONING_TEMPLATE.format(
            examples="\n".join(self.examples),
            question=self.question_template.format(question=question)
        )

        if step != 0:
            prompted_input += f"We have reached step {step}: " + '\n'
            steps_so_far = self.collect_steps(current_node).split('\n')
            for j, step_text in enumerate(steps_so_far):
                prompted_input += f"Step {j + 1}: " + step_text + '\n'
            prompted_input += "Now, please give the possible methods for the next step follow the steps before (if we have) (if previous step is a question or has a '?', you MUST answer or solve it, solve it, calculate it!!!):" + '\n' + f"Step {step + 1}: " + '\n'
        else:
            prompted_input += "Now, please give the possible methods for the first step:" + '\n' + f"Step {step + 1}: " + '\n'

        return prompted_input

    def collect_steps(self, current_node):
        steps = []
        node = current_node
        while node.parent is not None:
            steps.append(node.step)
            node = node.parent
        steps.reverse()
        return '\n'.join(steps)

    def handle_rethink(self, current_node, steps_now, question, tree, children_nodes, step):
        """
        Pruning operations in a DFS (Depth-First Search) structure.
        """
        flag = 0
        num = 0

        tree.remove_node(children_nodes[0])
        for child in current_node.children:
            res = ''
            while(res == ''):
                res = self.two_step_check_answer(question, steps_now + '\n' + child.step + '\n')
            if 'answer' in res:
                flag = 1
                break
            elif 'rethink' in res:
                num += 1
            else:
                flag = 2
                break

        for _ in range(num):
            tree.remove_node(current_node.children[0])

        if flag == 1:
            current_node = current_node.children[0]
        elif flag == 2:
            current_node = current_node.children[0]
            step += 1
        else:
            if current_node.parent:
                tmp_parent = current_node.parent
                tree.remove_node(current_node)
                if tmp_parent and tmp_parent.children:
                    current_node = tmp_parent.children[0]
                else:
                    step -= 1
                    current_node = tmp_parent
            else:
                step = 0

        return current_node, step, flag, res

    def construct_prediction(self, current_node, final_result, tree):
        prediction = ""

        """
        You can use the following code to view the steps that are still on the answer tree.
        """
        # def dfs(node):
        #     result = f"{node.step}\n"
        #     for child in node.children:
        #         result += dfs(child)
        #     return result

        # prediction += dfs(tree.root)

        steps = self.collect_steps(current_node).split('\n')
        
        for i, step in enumerate(steps):
            if step:
                prediction += f"{step}\n"
        prediction += final_result
        
        return prediction

    def do_reasoning(self, question,context=None):
        tree = Tree()
        current_node = tree.root
        tot_fault = 0
        i = 0

        while i < self.max_step:
            # prompted_input = self.construct_prompt(question, current_node, i)
            # responses = self.llm(prompted_input)
            # cleaned_methods = self.extract_methods(responses, i)
            # steps_now = self.collect_steps(current_node)
            # ranks = self.score_methods(question, steps_now, cleaned_methods)

            # if not self.is_valid_ranking(ranks, len(cleaned_methods)):
            #     ranks = list(range(len(cleaned_methods)))

            # steps = [f'Step {i + 1}: ' + cleaned_methods[rank] for rank in ranks]
            children_nodes=[]
            while(children_nodes==[]):
                prompted_input = self.construct_prompt(question, current_node, i)
                responses = self.llm(prompted_input)
                cleaned_methods = self.extract_methods(responses, i)
                steps_now = self.collect_steps(current_node)
                ranks = self.score_methods(question, steps_now, cleaned_methods)

                if not self.is_valid_ranking(ranks, len(cleaned_methods)):
                    ranks = list(range(len(cleaned_methods)))

                steps = [f'Step {i + 1}: ' + cleaned_methods[rank] for rank in ranks]
                children_nodes = tree.add_steps(current_node, steps)
            
            res = self.two_step_check_answer(question, steps_now + '\n' + children_nodes[0].step + '\n')

            flag = 0

            if 'rethink' in res:
                tot_fault += 1
                current_node, i, flag, res = self.handle_rethink(current_node, steps_now, question, tree, children_nodes, i)
            elif 'answer' in res:
                steps_now += '\n' + children_nodes[0].step + '\n'
                current_node = current_node.children[0]
                break
            else:
                steps_now += '\n' + children_nodes[0].step + '\n'
                current_node = children_nodes[0]
                i += 1

            if flag == 1:
                break

            if tot_fault >= 10:
                break
        tree.print_all_nodes()
        image=tree.draw_tree()
        return (self.construct_prediction(current_node, res, tree), image)