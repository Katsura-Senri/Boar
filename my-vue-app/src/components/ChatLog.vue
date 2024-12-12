<template>
  <div id="chat-log">
    <div v-for="(message, index) in chatLog" :key="index" :class="['message', message.sender]">
      <p>
        <span class="sender">
          {{ message.sender === 'user' ? 'You' : 'Boar' }}
        </span>
        {{ message.text }} <!-- 显示文本内容 -->
        
      </p>
      <!-- 使用 v-html 渲染图表的 HTML 内容 -->
      <!-- <div v-if="message.image"  class="image-container">
        111
        <template v-html="message.image">

        </template>
      </div> -->
      <template v-if="url != '' && message.sender === 'bot' && message.fileName">
        <iframe :src="message.fileName" frameborder="0" style="height: 100vh;"></iframe>
      </template>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    chatLog: {
      type: Array,
      required: true
    }
  },
  data() {
    return {
      url: "",
    }
  },
  watch: {
    chatLog: {
      deep: true,
      handler() {
        this.url = ""
        if (this.chatLog[this.chatLog.length - 1].text != "I'm thinking about it...") {
          this.Url()
        }
      }
    }
  },
  methods: {
    Url() {
      let idx
      for (let i = 0; i < this.chatLog.length; i++) {
        if (this.chatLog[i].sender === "bot") {
          idx = i
          break
        }
      }
      const blob = new Blob([this.chatLog[idx].image], { type: 'text/html' })
      this.url = URL.createObjectURL(blob)
    }
  },
};
</script>

<style scoped>
#chat-log {
  display: flex;
  flex-direction: column;
  height: 565px;
  overflow-y: auto;
  background-color: rgb(216.8, 235.6, 255);
  padding: 20px;
  border-radius: 20px;
  margin-bottom: 20px;
  flex: 1;
}

.message {
  padding: 10px 20px;
  border-radius: 30px;
  margin: 8px 0;
  max-width: 75%;
  word-wrap: break-word;
  font-size: 18px;
  display: flex;
  flex-direction: column;
}

.sender {
  font-weight: bold;
  margin-bottom: 5px;
}

.user {
  align-self: flex-end;
  background-color: rgb(250, 236.4, 216);
  color: #303133;
  margin-left: auto;
  text-align: left;
}

.bot {
  align-self: flex-start;
  background-color: rgb(224.6, 242.8, 215.6);
  color: #303133;
  text-align: left;
}

.image-container {
  margin-top: 10px;
}

.image-container img {
  max-width: 100%;
  height: auto;
}
</style>