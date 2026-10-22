#需要时刻知道当前程序做了什么（日志）
import logging
import os
from datetime import datetime
from utils.path_tool import get_abs_path

#日志保存的根目录
LOG_ROOT=get_abs_path("logs")#以后的日志都在工程文件里的logs文件夹内

#确保日志的目录存在
os.makedirs(LOG_ROOT,exist_ok=True)#exist_ok=True表示若目录不存在则创建

#日志的输出格式配置(基础)
DEFAULT_LOG_FORMAT=logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)#时间-名字-级别-文件名（）哪一行-正文

#日志管理器
def get_logger(
        name:str="agent",
        console_level:int=logging.INFO,
        file_level:int=logging.DEBUG,
        log_file=None,
)->logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)#默认设置DEBUG，可以改

    #避免重复添加handler
    if logger.handlers:#缺少的话会重复打印日志（如果已经存在就不要重复执行添加了，避免日志重复输出）
        return logger

    #控制台的handler
    coonsole_handler=logging.StreamHandler()#控制台的处理器(StreamHandler:将日志消息发送到任何支持流式写入的对象，最常见的就是在控制台打印日志)
    coonsole_handler.setLevel(console_level)
    coonsole_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(coonsole_handler)#添加控制台(Logger对象的方法，作用是给日志记录器添加一个处理器（handler），从而决定日志的输出目标（控制台、文件、网络等）)

    #文件handler
    if not log_file:
        log_file=os.path.join(LOG_ROOT,f"{name}_{datetime.now().strftime('%Y%m%d')}.log")#此处是绝对路径

    file_handler=logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)#基础信息格式

    logger.addHandler(file_handler)#到这，logger对象已经有两个handler，一个控制台一个文件

    return logger

#快捷获取日志管理器
logger=get_logger()#使用的时候直接import这个logger就行

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
    #不输出是因为INFO（控制台那）级别为20，只有大于他的才输出，而DEBUG是10