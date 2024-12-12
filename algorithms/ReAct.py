import requests
import json
import random
from .utils import Env, REACT_TEMPLATE
from .base import Algorithm
import networkx as nx
import plotly.graph_objs as go
from plotly.io import to_html
import textwrap
from fileCount import setFileCount, getFileCount

import networkx as nx
import plotly.graph_objs as go
import textwrap
import re

def draw_graph(sentences):
    G = nx.DiGraph()
    
    # 解析句子并构建链表结构
    def add_nodes_and_edges(sentences):
        previous_thought = None
        
        for i, sentence in enumerate(sentences):
            # 提取索引和内容
            parts = sentence.split(": ", 1)
            if len(parts) == 2:
                index, content = parts
                node_id = index
                label_text = f"{node_id}: " + "\n".join(textwrap.wrap(content, width=20))
                G.add_node(node_id, label=label_text)
                
                # 连接当前节点和前一个节点
                if previous_thought:
                    G.add_edge(previous_thought, node_id)
                
                # 更新前一个节点
                previous_thought = node_id
    
    add_nodes_and_edges(sentences)
    
    # 打印图的节点和边
    # print("Nodes:", G.nodes(data=True))
    # print("Edges:", G.edges())
    
    # 计算节点位置
    pos = {}
    for i, node in enumerate(G.nodes):
        pos[node] = (i, 0)  # 所有节点都在同一行，水平排列
    
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
        text=[G.nodes[node]['label'].replace("\n", "<br>") for node in G.nodes],
        textposition='top center',
        hoverinfo='text'
    )

    # 创建图表布局
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Linked List Structure with Hover",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    # 显示图表
    # fig.show()
    
    # 生成 HTML 文件
    setFileCount()
    fig.write_html(f"my-vue-app/public/test{getFileCount()}.html")
    
    # 获取 HTML 字符串
    html_str = to_html(fig, full_html=True, include_plotlyjs=False)
    html_f = '<!DOCTYPE html>' + html_str
    
    return html_f

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

def get_question_template(self):
        # formulate the input
        template = "Q: {question}\nA:"
        return template
import re

def split_sentence(sentence):
    print(sentence)
    """
    将句子按照 'Action {i}', 'Thought {i}', 'Observation {i}' 分割成多个小句。
    
    :param sentence: 需要被分割的原始句子
    :return: 包含多个小句的列表
    """
    # 定义正则表达式模式，用于匹配分隔符
    pattern = r'(Thought \d+):'
    
    # 使用正则表达式的 split 方法分割句子
    parts = re.split(pattern, sentence)
    
    # 初始化一个空列表来保存结果
    result = []
    
    # 遍历所有部分
    for i in range(1, len(parts), 2):
        # 每个小句由匹配项和其后的内容组成
        result.append(f"{parts[i]}: {parts[i+1].strip()}")
    
    # 返回结果列表
    return result

class ReAct(Algorithm):
    """
    ReAct prompt:
    including ReAct
    """
    def __init__(self, method='react'):
        self.MAX_ATTEMPTS = 10
        self.method = method
        self.prompt_name = method + '_prompt'
        self.examples= get_example(self, shots=5, algorithm = 'react_prompt')
        self.env = Env()
        self.question_template=get_question_template(self)

    def step(self, action):
        attempts = 0
        while attempts < self.MAX_ATTEMPTS:
            try:
                return self.env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1

    def do_reasoning(self, question, context, rounds=10):
        thoughts = []
        actions = []
        observations = []
        step_strs = []

        inputs = REACT_TEMPLATE.format(examples="\n".join(self.examples), question=self.question_template.format(question=question))

        n_calls, n_badcalls = 0, 0
        done = False
        
        for i in range(1, rounds):
            thought_action = self.model.generate(
                inputs+f"Thought {i}:", 
                stop=[f"\nObservation ", "assistant"]
            ).text[0]
            
            n_calls += 1
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except ValueError:
                try:
                    action = thought_action.strip().split("Action ")[1]
                    action = action[3:]
                    thought = "I need to " + action
                except IndexError:
                    try:
                        thought = thought_action.strip().split("\nAction ")[1].split("Thought ")[-1]
                        action = "I need to do something..."
                    except:
                        thought = thought_action.strip().split('\n')[0]
                        action = self.model.generate(inputs+f"Thought {i}:", stop=[f"\n"]).text[0].strip()
            
            obs, r, done, info = self.step(action)
            obs = obs.replace('\\n', '')
            action = action.replace(f'Thought {i}:','')
            action = action.replace(f'Observation {i}:','')
            obs = obs.replace(f'Thought {i}:', '')
            obs = obs.replace(f'Action {i}:', '')
            if "Thought" in thought:
                step_str = f"{thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            else:
                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            step_strs.append(step_str)
            inputs += step_str
            
            if done:
                break
        
        if not done:
            obs, r, done, info = self.step("finish[0]")
            
        result = "".join(step_strs)
        image_sentence = split_sentence(result)
        image_str = draw_graph(image_sentence)
        return (result,image_str)
