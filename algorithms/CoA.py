import nltk
from transformers import AutoTokenizer
from .base import Algorithm


class CoA(Algorithm):
    def __init__(self, method='coa'):
        self.method = method
        self.prompt_name = method + '_prompt'
        self.__tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.__dataset = None
        self.__requirement = None
        self.__supported_datasets = [dataset.lower() for dataset in
                                     ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
                                      "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum",
                                      "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en",
                                      "passage_retrieval_zh", "lcc", "repobench-p"]]
        # self.__requirement = "Read the following context carefully and answer the question as precisely as possible and trust only the information in the context. Do not provide reasoning for the answers. Output the answer by itself without any prefixes. Look for specific details, facts, or arguments that directly address the question. If you truly cannot find any relevant context, say 'I don't know'."

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset):
        if dataset.lower() not in self.__supported_datasets:
            raise ValueError(f"Dataset {dataset} is not supported. CoA supports long-context reasoning datasets.")
        self.__dataset = dataset

    def set_question_template(self, question_template):
        self.__requirement = question_template.split('\n')[0]

    def do_reasoning(self, question, context, rounds=0):
        if not context:
            raise ValueError("Context is required for CoA.")
        print(111)
        # print(f"\n########### QUESTION ##########\n\n {question}")
        list_chunks = self.__chunk_source_text(source_text=context,
                                               context_limit=self.model.max_tokens)
        print(222)
        # Workers process each chunk
        CU_prev = ""
        for chunk in list_chunks:
            output = self.model.generate(
                prompt=self.__get_worker_prompt(chunk=chunk, CU_prev=CU_prev, question=question),
                temperature=0,
                use_chat_api=True,
            ).text[0]
            # print(f"\n########### WORKER OUTPUT ##########\n\n {output}")
            CU_prev = output
        print(333)
        # Manager generates context
        final_answer = self.model.generate(
            prompt=self.__get_manager_prompt(CU_last=CU_prev, question=question),
            temperature=0,
            use_chat_api=True,
        ).text[0]
        print(final_answer)
        # print(f"\n########### ANSWER ##########\n\n {final_answer}")
        return final_answer

    def __chunk_source_text(self, source_text: str, context_limit: int = 2048) -> list[str]:
        list_sents = nltk.sent_tokenize(source_text)
        list_chunks = []
        curr_chunk_tokens = 0
        curr_chunk_sents = []
        for sent in list_sents:
            curr_sent_tokens = len(self.__tokenizer.tokenize(sent))
            if curr_chunk_tokens + curr_sent_tokens > context_limit:
                list_chunks.append(' '.join(curr_chunk_sents))
                curr_chunk_sents = [sent]
                curr_chunk_tokens = curr_sent_tokens
            else:
                curr_chunk_sents.append(sent)
                curr_chunk_tokens += curr_sent_tokens

        if curr_chunk_tokens:
            list_chunks.append(' '.join(curr_chunk_sents))

        return list_chunks

    def __get_worker_prompt(self, chunk: str, CU_prev: str, question: str) -> str:
        return f"""{chunk}
        
        Here is the summary of the previous source text: {CU_prev}
        
        Question: {question}
        
        You need to read current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to answer the Query, if any. So please write the summary that can include the evidence for answering the Query:"""

    def __get_manager_prompt(self, CU_last: str, question: str) -> str:
        return f"""{self.__requirement}
        
        The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
        
        {CU_last}
        
        Question: {question}
        
        Answer: """
