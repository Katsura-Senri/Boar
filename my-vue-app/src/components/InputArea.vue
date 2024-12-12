<template>
    <div class="input-area">
      <input class="text-input" v-model="userInput" placeholder="Please input your question." @keyup.enter="sendMessage"/>
      <button @click="sendMessage">发送</button>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        userInput: "",
      };
    },
    methods: {
      sendMessage() {
        this.$emit("send-message", this.userInput);
        this.userInput = "";
      },
    },
  };
  </script>

  <style scoped>
  .input-area {
    display: flex;
    gap:10px;
    margin-top: 20px;
    background-color: #ffffff;
  }

  .text-input {
    width: 900px;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 15px;
    font-size: 18px;
}
  
  button {
    padding: 8px 16px;
    border: none;
    border-radius: 10px;
    background-color: #409EFF;
    color: white;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }
  </style>
  