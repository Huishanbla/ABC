import os
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_embedding
from lightrag.llm import openai_complete_if_cache




# 模型全局参数配置  根据自己的实际情况进行调整
# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-FQZgr4fvjIv8iKaTR8QgtvEEhdS6CfFcNI1EHUTiVqD0R4hr"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# 非gpt大模型相关配置(oneapi方案 通义千问为例) 根据自己的实际情况进行调整
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_CHAT_API_KEY = "sk-0FxX9ncd0yXjTQF877Cc9dB6B2F44aD08d62805715821b85"
ONEAPI_CHAT_MODEL = "qwen-max"
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"
# 本地大模型相关配置(Ollama方案 llama3.1:latest为例) 根据自己的实际情况进行调整
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_CHAT_API_KEY = "ollama"
OLLAMA_CHAT_MODEL = "llama3.1:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"


WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        model=OPENAI_CHAT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=OPENAI_CHAT_API_KEY,
        base_url=OPENAI_API_BASE,
        **kwargs
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_CHAT_API_KEY,
        base_url=OPENAI_API_BASE,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func
    )
)


with open("./dickens/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())


# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )


# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )
#
# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )
