import os
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from datetime import datetime
import requests

rag=RagSummarizeService()
user_ids=["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1012","1013"]
external_data= {}
#缓存(全局缓存变量)
_LOCATION_CACHE=None

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query:str)->str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city:str)->str:
    try:
        response=requests.get(
            f"https://wttr.in/{city}?format=%C+%t+%w+%h",
            timeout=5
        )
        if response.status_code==200:
            weather_text=response.text.strip()
            return f"城市{city}的天气：{weather_text}"
    except Exception as e:
        logger.warning(f"获取天气失败：{e}")
    return f"获取天气失败，已记录在logging warning"

@tool(description="获取用户所在城市的名称，以纯字符串的形式返回")
def get_user_location()->str:
    global _LOCATION_CACHE
    if _LOCATION_CACHE:
        return _LOCATION_CACHE
    try:
        response=requests.get("http://ip-api.com/json/", timeout=3)
        if response.status_code==200:
            data=response.json()
            if data.get("status") == "success":
                city=data.get("city","深圳")
                _LOCATION_CACHE=city
                logger.info(f"检测到位置: {city}")
                return city
    except Exception as e:
        logger.warning(f"获取位置失败：{e}")
    _LOCATION_CACHE="深圳"
    return "深圳"

@tool(description="获取当前月份，以纯字符串的形式返回")
def get_current_month()->str:
    return datetime.now().strftime("%Y-%m")

@tool(description="获取用户的ID，以纯字符串的形式返回")
def get_user_id()->str:
    return random.choice(user_ids)

def generate_external_data():
    '''
    {
        "user_id": {
            "month":{"特征":xxx,"效率":xxx,...},
            ...
        },
        ...
    }
    :return:
    '''
    if not external_data:
        #如果里面没有东西
        external_data_path=get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):#如果不存在
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:#序列切片
                arr:list[str] = line.strip().split(",")#去掉前后空格
                user_id:str = arr[0].replace('"',"")#双引号替换成空字符串
                feature:str = arr[1].replace('"', "")
                efficiency:str=arr[2].replace('"',"")
                consumables:str=arr[3].replace('"',"")
                comparison:str=arr[4].replace('"',"")
                time:str=arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id]={}

                external_data[user_id][time]={
                    "特征":feature,
                    "效率":efficiency,
                    "耗材":consumables,
                    "对比":comparison,
                }

@tool(description="从外部系统中获取用户的使用记录，以纯字符串的形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id:str,month:str)->str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"