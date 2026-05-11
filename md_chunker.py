# 从 unstructured 导入 markdown 解析器
import json

from unstructured.partition.md import partition_md

# 导入 unstructured 中的 element 类型
from unstructured.documents.elements import (

    # 标题
    Title,

    # 普通正文
    NarrativeText,

    # 列表项
    ListItem,

    # 表格
    Table,

    # 代码块
    CodeSnippet,
)

# 导入类型注解
from typing import List, Dict

# 导入 Path 用于处理文件路径
from pathlib import Path

# 导入 uuid 用于生成 chunk 唯一ID
import uuid


# 定义 MarkdownChunker 类
class MarkdownChunker:
    """
    工业级 Markdown Chunker
    """

    # 初始化函数
    def __init__(

            self,

            # chunk 最大长度
            max_chunk_size: int = 1200,

            # chunk overlap 重叠长度
            overlap: int = 150,
    ):

        # 保存最大chunk大小
        self.max_chunk_size = max_chunk_size

        # 保存 overlap 大小
        self.overlap = overlap

    # 加载 markdown 文件
    def load_markdown(self, file_path: str):

        # 使用 unstructured 解析 markdown
        elements = partition_md(filename=file_path)

        # 返回解析后的 element 列表
        return elements

    # 创建 chunk
    def create_chunk(

            self,

            # 文本列表
            text_list,

            # 标题路径
            title_path,

            # 内容类型
            content_type,

            # 来源文件
            source,
    ):

        # 把 text_list 合并成完整文本
        text = "\n".join(text_list).strip()

        # 返回 chunk 字典
        return {

            # chunk 唯一ID
            "chunk_id": str(uuid.uuid4()),

            # 来源文件名
            "source": source,

            # 标题路径
            "title_path": title_path.copy(),

            # 内容类型
            "content_type": content_type,

            # chunk 文本
            "text": text,

            # 字符数
            "char_count": len(text),
        }

    # 长文本切分函数
    def split_large_text(self, text: str):

        """
        长文本语义切分
        """

        # 如果文本长度没有超过限制
        if len(text) <= self.max_chunk_size:
            # 直接返回
            return [text]

        # 存储切分结果
        chunks = []

        # 起始位置
        start = 0

        # 循环切分
        while start < len(text):
            # 当前chunk结束位置
            end = start + self.max_chunk_size

            # 截取chunk
            chunk = text[start:end]

            # 保存chunk
            chunks.append(chunk)

            # 更新start
            # 使用 overlap 实现上下文重叠
            start = end - self.overlap

        # 返回切分后的 chunks
        return chunks

    # 主 chunk 切分逻辑
    def chunk_elements(

            self,

            # unstructured elements
            elements,

            # 文件来源
            source,
    ) -> List[Dict]:

        # 最终 chunk 结果
        chunks = []

        # 标题栈
        # 用于保存层级标题路径
        title_stack = []

        # 当前 chunk 文本列表
        current_texts = []

        # 当前 chunk 大小
        current_size = 0

        # flush函数
        # 用于保存当前chunk
        def flush_current_chunk():

            # 使用外层变量
            nonlocal current_texts

            # 使用外层变量
            nonlocal current_size

            # 如果当前没有内容
            if not current_texts:
                # 直接返回
                return

            # 创建chunk
            chunk = self.create_chunk(

                # 当前文本
                text_list=current_texts,

                # 当前标题路径
                title_path=title_stack,

                # 内容类型
                content_type="text",

                # 来源文件
                source=source,
            )

            # 保存chunk
            chunks.append(chunk)

            # 清空当前文本
            current_texts = []

            # 重置大小
            current_size = 0

        # 遍历所有 element
        for el in elements:

            # 获取文本并去除空格
            text = el.text.strip()

            # 空文本跳过
            if not text:
                continue

            # ====================
            # TITLE
            # ====================

            # 如果是标题
            if isinstance(el, Title):
                # 先保存旧chunk
                flush_current_chunk()

                # 简化版标题层级
                # 实际工业级需要结合 metadata.level
                title_stack.append(text)

                # 标题本身不进入chunk
                continue

            # ====================
            # TABLE
            # ====================

            # 如果是表格
            if isinstance(el, Table):
                # 先保存旧chunk
                flush_current_chunk()

                # 创建table chunk
                table_chunk = self.create_chunk(

                    # 表格作为单独chunk
                    text_list=[text],

                    # 当前标题路径
                    title_path=title_stack,

                    # 内容类型
                    content_type="table",

                    # 来源文件
                    source=source,
                )

                # 保存table chunk
                chunks.append(table_chunk)

                # 继续下一轮
                continue

            # ====================
            # CODE
            # ====================

            # 如果是代码块
            if isinstance(el, CodeSnippet):

                # 先flush当前普通文本
                flush_current_chunk()

                # 对超长代码做切分
                code_chunks = self.split_large_text(text)

                # 遍历代码chunk
                for code in code_chunks:
                    # 创建代码chunk
                    chunk = {

                        # chunk ID
                        "chunk_id": str(uuid.uuid4()),

                        # 来源文件
                        "source": source,

                        # 标题路径
                        "title_path": title_stack.copy(),

                        # 内容类型
                        "content_type": "code",

                        # 代码文本
                        "text": code,

                        # 字符数
                        "char_count": len(code),
                    }

                    # 保存chunk
                    chunks.append(chunk)

                # 继续下一轮
                continue

            # ====================
            # NORMAL TEXT
            # ====================

            # 如果是普通文本或列表
            if isinstance(el, (NarrativeText, ListItem)):

                # 如果单段文本已经超长
                if len(text) > self.max_chunk_size:

                    # 先flush旧chunk
                    flush_current_chunk()

                    # 对长文本做切分
                    split_chunks = self.split_large_text(text)

                    # 遍历切分结果
                    for split_text in split_chunks:
                        # 创建chunk
                        chunk = self.create_chunk(

                            # 当前切分文本
                            text_list=[split_text],

                            # 标题路径
                            title_path=title_stack,

                            # 内容类型
                            content_type="text",

                            # 来源文件
                            source=source,
                        )

                        # 保存chunk
                        chunks.append(chunk)

                    # 跳过后续逻辑
                    continue

                # 如果当前chunk放不下
                if current_size + len(text) > self.max_chunk_size:
                    # 保存旧chunk
                    flush_current_chunk()

                # 加入当前chunk
                current_texts.append(text)

                # 更新chunk大小
                current_size += len(text)

        # 循环结束后
        # flush最后一个chunk
        flush_current_chunk()

        # 返回所有chunk
        return chunks

    # 完整处理流程
    def process(self, file_path: str):

        # 获取文件名
        source = Path(file_path).name

        # 加载markdown
        elements = self.load_markdown(file_path)

        # 执行chunk切分
        chunks = self.chunk_elements(

            # elements
            elements=elements,

            # 来源文件
            source=source,
        )

        # 返回chunks
        return chunks

    # 存储到json方便调试
    def save_chunks_to_json(self, chunks, output_path="debug_chunks.json"):
        """
        保存 chunk 到 json 文件
        """

        # 写入 json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,

                # 保留中文
                ensure_ascii=False,

                # 美化缩进
                indent=2
            )

        print(f"Chunk 已保存到: {output_path}")


# 程序入口
if __name__ == "__main__":

    # 创建 chunker 实例
    chunker = MarkdownChunker(

        # 最大chunk长度
        max_chunk_size=1200,

        # overlap长度
        overlap=150,
    )

    # 处理 markdown 文件
    chunks = chunker.process("data/test.md")

    # 保存 json debug 文件
    chunker.save_chunks_to_json(
        chunks,
        "output/md_chunks.json"
    )

    # 打印分隔线
    print("=" * 100)

    # 打印chunk总数
    print(f"TOTAL CHUNKS: {len(chunks)}")

    # 打印分隔线
    print("=" * 100)

    # 遍历所有chunk
    for i, chunk in enumerate(chunks):
        # 打印chunk编号
        print(f"\nCHUNK {i + 1}")

        # 打印分隔线
        print("-" * 100)

        # 打印标题路径
        print("TITLE PATH:")

        # 拼接标题路径
        print(" > ".join(chunk["title_path"]))

        # 打印内容类型
        print("\nCONTENT TYPE:")

        # 输出内容类型
        print(chunk["content_type"])

        # 打印字符数
        print("\nCHAR COUNT:")

        # 输出字符数
        print(chunk["char_count"])

        # 打印正文
        print("\nTEXT:")

        # 只打印前1000字符
        print(chunk["text"][:1000])

        # 空行
        print("\n")
