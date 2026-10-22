from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_location,get_user_id,get_current_month,
                                     fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent=create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize,get_weather,get_user_location,get_user_id,get_current_month,
                   fetch_external_data,fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def execute_stream(self,query:str):
        input_dict={
            "messages":[
                {"role":"user","content":query}
            ]
        }
        #第三个参数context就是上下文runtime中的信息，就是做提示词切换的标记
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):#先让字典存在，才能改true or false
            latest_message=chunk["messages"][-1]#拿到最新一条消息
            if latest_message:
                yield latest_message.content.strip()+"\n"#yield:懒加载，含有 yield 的函数不再是普通函数，而是一个生成器函数，调用它会返回一个生成器对象

if __name__ == '__main__':
    agent=ReactAgent()
    for chunk in agent.execute_stream("给我生成我的试用报告"):
        print(chunk,end="",flush=True)