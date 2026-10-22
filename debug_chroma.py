import traceback
from rag.vector_store import VectorStoreService

print("=" * 50)
print("调试 ChromaDB 写入")
print("=" * 50)

vs = VectorStoreService()
print(f"集合名称: {vs.vector_store._collection.name}")
print(f"当前文档数: {vs.vector_store._collection.count()}")

try:
    print("\n开始加载文档...")
    vs.load_document()
    print(f"\n加载后文档数: {vs.vector_store._collection.count()}")
except Exception as e:
    print("\n!!! 加载失败，完整错误信息：")
    traceback.print_exc()