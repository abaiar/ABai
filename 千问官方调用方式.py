from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts import MessagesPlaceholder
import time
import asyncio
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate

chatLLM = ChatTongyi(
    
    model="qwen-flash",   # 此处以qwen-flash为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    streaming=False,
    enable_thinking=False,
    temperature=0,
    max_tokens=50,
    # other params...
)

 
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位乐于助人的智能小助手"),
    (MessagesPlaceholder(variable_name="history")),
    ("human", "{input}")
])



#新版调用方式
chain = prompt | chatLLM


# 流式输出
# for chunk in chatLLM.stream(messages):
#     print(chunk.content, end="", flush=True)


res =   chain.invoke({"input": "今天天气如何？"})
print(res.content)

