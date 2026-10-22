"""测试向量库清空/重置功能"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("测试向量库清空/重置功能")
print("=" * 60)

# ========== 测试1：查看当前状态 ==========
print("\n【测试1】查看当前知识库状态")
print("-" * 40)

from utils.streamlit_session import update_knowledge_base_status

doc_count = update_knowledge_base_status()
print(f"当前文档数: {doc_count}")

# ========== 测试2：清空向量库 ==========
print("\n【测试2】清空向量库（clear_vector_store）")
print("-" * 40)

from utils.streamlit_session import clear_vector_store

result = clear_vector_store()
if result is not None:
    print("✅ 清空成功")
else:
    print("❌ 清空失败")

# 再次查看状态
doc_count = update_knowledge_base_status()
print(f"清空后文档数: {doc_count}")

# ========== 测试3：完全重置 ==========
print("\n【测试3】完全重置（reset_vector_store）")
print("-" * 40)

from utils.streamlit_session import reset_vector_store

result = reset_vector_store()
if result is not None:
    print("✅ 重置成功")
else:
    print("❌ 重置失败")

# 再次查看状态
doc_count = update_knowledge_base_status()
print(f"重置后文档数: {doc_count}")

# ========== 测试4：重新加载文档 ==========
print("\n【测试4】重新加载文档（rebuild_vector_store）")
print("-" * 40)

from utils.streamlit_session import rebuild_vector_store

new_count = rebuild_vector_store()
print(f"重新加载后文档数: {new_count}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)