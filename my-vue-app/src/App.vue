<template>
  <div id="app">
    <h1 id="title">Hust Boar</h1>
    <div id="container">
      <aside id="notes">
        <div v-if="tooltip.visible" class="bubble">{{ tooltip.content }}</div>
      </aside>
      <aside id="button-side">
        <button
          :class="{ selected: selectedAlgorithm === 'Chat' }"
          @click="selectAlgorithm('Chat')"
          @mouseover="showTooltip('这是 Chat 模式。此模式适用于简单的聊天任务。')"
          @mouseleave="hideTooltip"
        >
          Chat
        </button>
        <button
          :class="{ selected: mathsMode }"
          @click="setMathsMode"
          @mouseover="showTooltip(mathsMode ? '点击以折叠 Math 模式' : '这是 Maths 模式. 使用数学算法执行任务')"
          @mouseleave="hideTooltip"
          class="maths-button"
        >
          {{ mathsMode ? 'Collapse' : 'Maths' }}
        </button>
        <div v-if="mathsMode" class="maths-buttons">
          <button
            v-for="algo in mathsAlgorithms"
            :key="algo"
            :class="{ selected: selectedAlgorithm === algo }"
            @mouseover="showTooltip(notes[algo])"
            @mouseleave="hideTooltip"
            @click="selectAlgorithm(algo)"
          >
            {{ algo }}
          </button>
        </div>
        <button id="clear-chat" @click="clearChatLog">Clear</button>
      </aside>
      <div id="chatbox">
        <ChatLog :chatLog="chatLog" />
        <InputArea @send-message="sendMessage" />
      </div>
    </div>
  </div>
</template>

<script>
import ChatLog from './components/ChatLog.vue';
import InputArea from './components/InputArea.vue';

export default {
  components: {
    ChatLog,
    InputArea,
  },
  data() {
    return {
      userInput: "",
      selectedAlgorithm: "Chat",
      algorithms: ["Chat", "CoT", "Standard", "ToT","ReAct"],
      chatLog: [],
      mathsMode: false,
      tooltip: {
        visible: false,
        content: "",
      },
      notes: {
        Chat: "这是 Chat 模式。此模式适用于简单的聊天任务。",
        CoT: "这是 CoT 模式。此模式适用于链式思维任务。",
        Standard: "这是 Standard 模式。此模式适用于常规的标准算法任务。",
        ToT: "这是 ToT 模式。此模式适用于任务分解和协作式任务。",
        ReAct:  "这是 ReAct 模式"
      },
    };
  },
  computed: {
    mathsAlgorithms() {
      return this.algorithms.slice(1);
    },
  },
  methods: {
    selectAlgorithm(algo) {
      this.selectedAlgorithm = algo;
      if (algo === 'Chat') {
        this.mathsMode = false;
      }
    },
    setMathsMode() {
      this.mathsMode = !this.mathsMode;
      if (!this.mathsMode) {
        this.selectedAlgorithm = 'Chat';
      }
    },
    showTooltip(content) {
      this.tooltip.visible = true;
      this.tooltip.content = content;
    },
    hideTooltip() {
      this.tooltip.visible = false;
    },
    clearChatLog() {
      this.chatLog = [];
    },
    async sendMessage(input) {
      if (!input.trim()) return;

      this.chatLog.push({ sender: "user", text: input });
      const loadingMessage = { sender: "bot", text: "I'm thinking about it..." };
      this.chatLog.push(loadingMessage);

      try {
        const response = await fetch("http://127.0.0.1:5000/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: "together_ai/meta-llama/Llama-3-8b-chat-hf",
            algo_name: this.selectedAlgorithm.toLowerCase(),
            question: input,
          }),
        });

        const data = await response.json();
        this.chatLog.pop();

        if (data && data.reply) {
          if (data.image) {
              this.chatLog.push({
              sender: "bot",
              text: data.reply,
              image: data.image,  // 将图表的 HTML 内容添加到 chatLog 中
              fileName: data.fileName
            });
          // console.log(this.chatLog)
      } else {
        this.chatLog.push({ sender: "bot", text: data.reply });
      }
        } else {
          this.chatLog.push({ sender: "bot", text: "服务器返回了错误的数据格式。" });
        }
      } catch (error) {
        this.chatLog.pop();
        this.chatLog.push({ sender: "bot", text: "出现错误 - " + error.message });
      }
    },
  },
};
</script>

<style scoped>
#app {
  width: 90vw;
  height: 90vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
}
#container {
  display: flex;
  justify-content: flex-end;
  position: fixed;
  right: 280px;
  top: 160px;
  bottom: 0;
  height:80vh;
  /* width: 90vw; */
  /* flex: 0.8; */
}

#title {
  text-align: center;
}

#notes {
  background-color:rgb(255, 255, 255);
  width: 180px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  overflow: hidden;
  white-space: normal;
  word-wrap: break-word;
}

.bubble {
  background-color: rgb(239.8, 248.9, 235.3);
  padding: 12px;
  border-radius: 20px;
  color: #333;
  font-size: 16px;
  margin-bottom: 5px;
  text-align: left;
  text-indent: 2ch;
}

#button-side {
  width: 100px;
  background-color: #ffffff;
  padding: 15px;
  display: flex;
  flex-direction: column;
}

.maths-button {
  margin-top: 30px;
  margin-bottom: 15px;
}

.maths-buttons {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

button {
  border: none;
  border-radius: 30px;
  height: 60px;
  cursor: pointer;
  background-color: #ffffff;
  transition: background-color 0.3s ease;
}

button.selected {
  background-color: #409EFF;
  color: white;
}

#clear-chat {
  margin-top: auto;
  padding: 10px;
  border: none;
  border-radius: 30px;
  background-color: #f56c6c;
  color: white;
  font-size: 16px;
  cursor: pointer;
}

#clear-chat:hover {
  background-color: #d9534f;
}


#chatbox {
  width: 1000px;
  margin: 0 auto;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 20px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  background-color: #fff;
  text-align: center;
  display: flex;
  flex-direction: column;
}
</style>
