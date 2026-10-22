import os
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

# 强制单线程，防止 Windows 内存报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()


def build_database():
    print("=" * 50)
    print("🚀 开始执行数据入库流程 (FAISS版)...")
    print("=" * 50)

    # 1. 读取文件
    with open("job_info.md", "r", encoding="utf-8") as f:
        markdown_document = f.read()

    # 2. 切片
    headers_to_split_on = [
        ("#", "H1_Main"),
        ("##", "H2_Section"),
        ("###", "H3_SubSection"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    final_chunks = text_splitter.split_documents(md_header_splits)
    print(f"[2/4] ✅ 切分完成！共生成 {len(final_chunks)} 个文本块。")

    # 3. 存入 FAISS 向量库
    print("\n[3/4] 🧠 正在调用智谱 API 构建 FAISS 向量库...")
    print("    (请稍候...)")
    try:
        embeddings = OpenAIEmbeddings(model="embedding-3")

        # 用 FAISS 构建
        db = FAISS.from_documents(final_chunks, embeddings)

        # 【关键】手动把 FAISS 索引保存到本地文件夹
        db.save_local("faiss_index")
        print("    ✅ FAISS 向量库构建并持久化成功！(已生成 faiss_index 文件夹)")
    except Exception as e:
        print(f"    ❌ 构建失败: {e}")
        return

    # 4. 构建 BM25
    print("\n[4/4] 🔤 正在构建 BM25 关键词索引...")
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("    ✅ BM25 索引构建成功！")

    print("\n" + "=" * 50)
    print("🎉 FAISS 数据库准备就绪！")
    print("=" * 50)


if __name__ == "__main__":
    build_database()