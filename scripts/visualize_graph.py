"""
LangGraph 架构图可视化脚
功能：
1. 导出 Graph 架构图（PNG/SVG）
2. 打印节点和边信息
3. 保存到指定目录
用法：
    python scripts/visualize_graph.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph_agent.graph import build_graph_with_persistence
import os

def export_graph_image(output_dir: str = None):
    """
    导出 Graph 架构图
    Args:
        output_dir: 输出目录，默认为项目根目录下的 assets/
    Returns:
        导出的文件路径列表
    """
    #设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "assets"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LangGraph 架构图导出")
    print("=" * 60)

    #构建 Graph
    print("\n【1】构建 Graph...")
    graph = build_graph_with_persistence()#加载 SQLite 持久化器，创建数据库连接
    print("✅ Graph 构建成功")
    #获取 Graph 结构
    print("\n【2】分析 Graph 结构...")
    graph_obj = graph.get_graph()#返回一个 Graph 对象，包含节点和边的结构信息（不是编译后的可执行图，而是纯结构描述）
    #打印节点信息
    nodes = list(graph_obj.nodes.keys())
    print(f"节点数量: {len(nodes)}")
    print(f"节点列表: {nodes}")
    #打印边信息(graph_obj.edges：边列表，每个边包含 source 和 target。)
    edges = graph_obj.edges
    print(f"   边数量: {len(edges)}")
    #导出图片
    print("\n【3】导出架构图...")
    exported_files = []

    #尝试导出 PNG
    try:
        png_data = graph_obj.draw_png()#draw_png():生成 PNG 图片的二进制数据
        png_path = output_dir / "agent_architecture.png"
        with open(png_path, "wb") as f:
            f.write(png_data)
        exported_files.append(str(png_path))
        print(f"   ✅ PNG: {png_path}")
    except Exception as e:
        print(f"   ⚠️ PNG 导出失败: {e}")
        print(f"      提示: 请安装 Graphviz (https://graphviz.org/download/)")

    # 尝试导出Mermaid格式（纯文本，无需依赖）
    '''
    Mermaid 格式示例：
        graph TD
        __start__ --> agent
        agent --> tools
        agent --> vacp
        tools --> agent
        vacp --> human_review
        human_review --> __end__
    '''
    try:
        mermaid_data = graph_obj.draw_mermaid()
        mermaid_path = output_dir / "agent_architecture.mmd"
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write(mermaid_data)
        exported_files.append(str(mermaid_path))
        print(f"   ✅ Mermaid: {mermaid_path}")
    except Exception as e:
        print(f"   ⚠️ Mermaid 导出失败: {e}")

    #尝试导出 ASCII 格式(draw_ascii()：在终端用 ASCII 字符画出架构图。)
    try:
        ascii_data = graph_obj.draw_ascii()
        print(f"\n【4】ASCII 架构图:\n")
        print(ascii_data)
    except Exception as e:
        print(f"   ⚠️ ASCII 导出失败: {e}")

    #保存节点和边信息为 JSON
    print("\n【5】保存结构信息...")
    import json
    structure = {
        "nodes": [str(n) for n in nodes],
        "edges": [
            {"source": str(e.source), "target": str(e.target)}
            for e in edges
        ]
    }
    # 等价于：
    # nodes_list = []
    # for n in nodes:
    #     nodes_list.append(str(n))
    #
    # edges_list = []
    # for e in edges:
    #     edges_list.append({"source": str(e.source), "target": str(e.target)})

    json_path = output_dir / "agent_structure.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)#ensure_ascii=False:允许中文直接显示，不转义/indent=2:缩进2个空格，美化输出
    exported_files.append(str(json_path))
    print(f"   ✅ JSON: {json_path}")

    print("\n" + "=" * 60)
    print(f"🎉 导出完成！共 {len(exported_files)} 个文件")
    print("=" * 60)

    return exported_files

def get_graph_structure():
    """
    获取 Graph 结构信息（供 Streamlit 调用）
    Returns:
        {
            "nodes": [...],
            "edges": [...],
            "mermaid": "...",
            "png_path": "..."
        }
    给Streamlit前端调用，返回结构化的字典
    """
    graph = build_graph_with_persistence()
    graph_obj = graph.get_graph()

    structure = {
        "nodes": list(graph_obj.nodes.keys()),
        "edges": [
            {"source": e.source, "target": e.target}#e.source和e.target没有str()转换（上一个函数有），Streamlit能直接处理
            for e in graph_obj.edges
        ]
    }
    #尝试获取 Mermaid
    try:
        structure["mermaid"] = graph_obj.draw_mermaid()
    except:
        structure["mermaid"] = None

    #尝试获取PNG路径(检查PNG文件是否存在，存在则返回路径，否则返回 None)
    png_path = Path(__file__).parent.parent / "assets" / "agent_architecture.png"
    if png_path.exists():
        structure["png_path"] = str(png_path)
    else:
        structure["png_path"] = None
    return structure

def print_graph_structure():
    """打印 Graph 结构（用于调试）"""
    structure = get_graph_structure()

    print("=" * 60)
    print("Graph 结构信息")
    print("=" * 60)

    print(f"\n节点 ({len(structure['nodes'])} 个):")
    for node in structure["nodes"]:
        print(f"  - {node}")

    print(f"\n边 ({len(structure['edges'])} 条):")
    for edge in structure["edges"]:
        print(f"  {edge['source']} → {edge['target']}")

    if structure.get("mermaid"):
        print(f"\nMermaid 格式:")
        print(structure["mermaid"])

if __name__ == "__main__":
    # 导出图片
    export_graph_image()
    # 打印结构
    print("\n")
    print_graph_structure()