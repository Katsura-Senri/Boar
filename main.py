import json
from graphviz import Digraph
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
from io import BytesIO
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from fire import Fire
from algorithms import *
from benchmarks import *
from models import *
import plotly.graph_objs as go
from plotly.io import to_html
from fileCount import setFileCount, getFileCount

app = Flask(__name__)
CORS(app)

def delete_html_files(directory):
    # 检查目标目录是否存在
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    # 遍历目录及其子目录，查找并删除所有 .html 文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def draw_chain(chain):
    G = nx.DiGraph()
    
    # 添加节点和边
    print(len(chain))
    for i in range(len(chain) - 1):
        G.add_node(i, label=chain[i])
        G.add_edge(i, i + 1)
    G.add_node(len(chain) - 1, label=chain[-1])
    
    # 计算节点位置
    pos = {i: (i, 0) for i in range(len(chain))}
    
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
                        title="Chain Structure with Hover",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    setFileCount()
    fig.write_html(f"my-vue-app/public/test{getFileCount()}.html")
    # 将图表转换为 HTML 字符串
    # fig.show()
    html_str = to_html(fig, full_html=True, include_plotlyjs=False)
    html_f = '<!DOCTYPE html>' + html_str
    return html_f

# 示例用法
# chain = ["Node A", "Node B", "Node C", "Node D"]
# html_output = draw_chain(chain)
# print(html_output)



@app.route('/run', methods=['POST'])

def run(model_name: str = 'together_ai/meta-llama/Llama-3-8b-chat-hf', algo_name: str = 'react'):
    """处理来自前端的请求，运行模型并返回生成的回复"""
    data = request.json
    question = data.get("question") 
    algo_name = data.get("algo_name")
    if  algo_name == 'chat':
        model_id = "together_ai/meta-llama/Llama-3-8b-chat-hf"  # 使用合适的模型ID
        model = opensource_API_models(model=model_id, max_tokens=256)  # 实例化自定义模型类
        messages = [
        {"role": "system", "content": "You are a helpful chatbot"},
        {"role": "user", "content": question}, ]
        response = model.generate_with_messages(messages)
        return jsonify({"reply": response.text[0]})
    model=opensource_API_models(model=model_name.split("together_ai/")[-1], max_tokens=500)
    # examples = dataset_class.get_example(shots=shots, algorithm=algorithm.prompt_name)
    # algorithm.set_example(examples)
    # 初始化模型和算法
    if algo_name == 'cot':
        algorithm = Prompt(method=algo_name)
    elif algo_name == 'standard':
        algorithm = Prompt(method=algo_name)
    elif algo_name == 'tot':
        algorithm = ToT(method=algo_name)
    else:
        algorithm = ReAct(method=algo_name)
    # elif algo_name == 'mcts':
    #     algorithm = RPMAD(method=algo_name)
    # algorithm = ToT(method=algo_name)
    algorithm.set_model(model)
    context = None
    # 执行推理生成回复
    image_str = None
    
    if algo_name == 'tot' or algo_name == 'react':
        (raw_pred, image_str)= algorithm.do_reasoning(question,context=context)
        # print(image_str)
    elif algo_name == 'standard':
        raw_pred = algorithm.do_reasoning(question,context=context)
        image_str=None
    elif algo_name == 'cot':
        raw_pred = algorithm.do_reasoning(question,context=context)
        pattern = r'(Step \d+: )'
        steps = re.split(pattern, raw_pred)
        steps = [steps[i] + steps[i+1] for i in range(1, len(steps), 2)]
        print(steps)
        image_str=draw_chain(steps)    
        # 处理分割结果
        # 去掉空字符串，并重新组合每个步骤
            
    response = {
        "reply": raw_pred,
        "image":image_str,
        "fileName": f"test{getFileCount()}.html"
    }
     
    return jsonify(response)

if __name__ == "__main__":
    directory = 'my-vue-app\public'
    delete_html_files(directory)
    app.run(port=5000)
