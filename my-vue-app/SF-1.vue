<template>
  <div id="app">
    <h1>对话机器人</h1>
    <div id="chatbox">
      <div id="chat-log">
        <div v-for="(message, index) in chatLog" :key="index" :class="message.sender">
          <p>{{ message.sender === 'user' ? '你' : '机器人' }}: {{ message.text }}</p>
        </div>
      </div>
      
      <!-- 算法选择按钮 -->
      <div class="algorithm-buttons">
        <button
          v-for="algo in algorithms"
          :key="algo"
          :class="{ selected: selectedAlgorithm === algo }"
          @click="selectAlgorithm(algo)"
        >
          {{ algo }}
        </button>
      </div>
      
      <!-- 用户输入框和发送按钮 -->
      <div class="input-area">
        <input v-model="userInput" placeholder="输入你的问题..." @keyup.enter="sendMessage" />
        <button @click="sendMessage">发送</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      userInput: "",
      selectedAlgorithm: "CoT",
      algorithms: ["CoT", "Standard"],
      chatLog: [],
    };
  },
  methods: {
    selectAlgorithm(algo) {
      this.selectedAlgorithm = algo;
    },
    async sendMessage() {
      if (!this.userInput.trim()) return;

      // 添加用户消息到聊天记录
      this.chatLog.push({ sender: "user", text: this.userInput });
      const loadingMessage = { sender: "bot", text: "正在思考..." };
      this.chatLog.push(loadingMessage);

      try {
        const response = await fetch("http://127.0.0.1:5000/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: "together_ai/meta-llama/Llama-3-8b-chat-hf",
            algo_name: this.selectedAlgorithm.toLowerCase(),
            question: this.userInput,
          }),
        });

        const data = await response.json();
        this.chatLog.pop(); // 移除加载状态

        if (data && data.reply) {
          this.chatLog.push({ sender: "bot", text: data.reply });
        } else {
          this.chatLog.push({ sender: "bot", text: "服务器返回了错误的数据格式。" });
        }
      } catch (error) {
        this.chatLog.pop();
        this.chatLog.push({ sender: "bot", text: "出现错误 - " + error.message });
      }

      this.userInput = "";
    },
  },
};
</script>

<style scoped>
#app {
  text-align: center;
  font-family: Arial, sans-serif;
}

#chatbox {
  max-width: 500px;
  margin: 0 auto;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

#chat-log {
  max-height: 400px;
  overflow-y: auto;
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 10px;
}

#chat-log p {
  margin: 5px 0;
}

.user p {
  text-align: left;
  color: #007bff;
}

.bot p {
  text-align: left;
  color: #333;
}

.algorithm-buttons {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 10px;
}

.algorithm-buttons button {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  background-color: #eee;
  transition: background-color 0.3s ease;
}

.algorithm-buttons button.selected {
  background-color: #007bff;
  color: white;
}

.input-area {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}

input[type="text"] {
  flex: 1;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

button {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #0056b3;
}
</style>
