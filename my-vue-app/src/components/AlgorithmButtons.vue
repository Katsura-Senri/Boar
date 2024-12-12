<template>
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
  </template>
  
  <script>
  export default {
    props: {
      algorithms: Array,
      selectedAlgorithm: String,
    },
    methods: {
      selectAlgorithm(algo) {
        this.$emit("select-algorithm", algo);
      },
    },
  };
  </script>
  
  <style scoped>
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
  </style>
  