'''
为整个工程提供统一的绝对路径
'''
import os

def get_project_root()->str:
    '''
    获取整个工程所在的的根目录
    :return: 字符串根目录
    '''
    # 当前文件的绝对路径
    current_file=os.path.abspath(__file__)
    #跳到当前文件所在的文件夹的绝对路径
    current_dir=os.path.dirname(current_file)
    #获取工程的根目录
    project_root=os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path:str)->str:
    '''
    给相对路径，返回绝对路径
    :param relative_path: 相对路基
    :return: 绝对路径
    '''
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)

if __name__=='__main__':
    print(get_abs_path("config/config.txt"))