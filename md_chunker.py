# 从 markdown-it-py 导入 MarkdownIt 解析器
import json

from markdown_it import MarkdownIt

# 导入 Token 类型
from markdown_it.token import Token

# 导入类型注解
from typing import List, Dict

# 导入 Path 用于文件读取
from pathlib import Path

# 导入 uuid 用于生成 chunk 唯一ID
import uuid


# 定义 Markdown AST Chunker 类
class MarkdownASTChunker:

    # 初始化函数
    def __init__(

            self,

            # chunk 最大长度
            max_chunk_size: int = 1500,

            # overlap 重叠长度
            overlap: int = 200,
    ):

        # 保存最大chunk大小
        self.max_chunk_size = max_chunk_size

        # 保存 overlap 大小
        self.overlap = overlap

        # 初始化 markdown-it parser
        self.md = MarkdownIt("commonmark", {

            # 禁止 HTML
            "html": False,

            # 自动识别链接
            "linkify": True,

            # 禁止智能符号替换
            "typographer": False,
        })

    # 解析 markdown
    def parse_markdown(

            self,

            # markdown 文本
            markdown_text: str,
    ) -> List[Token]:

        # 使用 markdown-it 解析 markdown
        tokens = self.md.parse(markdown_text)

        # 返回 token 列表
        return tokens

    # 长文本切分
    def split_large_text(

            self,

            # 待切分文本
            text: str,
    ):

        # 如果文本长度小于最大chunk
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

            # 更新起始位置
            # overlap 用于上下文重叠
            start = end - self.overlap

        # 返回chunks
        return chunks

    # 创建chunk
    def create_chunk(

            self,

            # chunk文本
            text: str,

            # 标题路径
            title_path: List[str],

            # 内容类型
            content_type: str,

            # 来源文件
            source: str,
    ):

        # 返回chunk字典
        return {

            # chunk唯一ID
            "chunk_id": str(uuid.uuid4()),

            # 来源文件
            "source": source,

            # 标题路径
            "title_path": title_path.copy(),

            # 内容类型
            "content_type": content_type,

            # chunk文本
            "text": text.strip(),

            # chunk字符数
            "char_count": len(text),
        }

    # 核心chunk构建逻辑
    def build_chunks(

            self,

            # markdown token列表
            tokens: List[Token],

            # 来源文件
            source: str,
    ):

        # 最终chunk结果
        chunks = []

        # 标题栈
        # 用于保存标题层级路径
        title_stack = []

        # 当前文本buffer
        current_text = []

        # 当前chunk大小
        current_size = 0

        # token索引
        i = 0

        # flush函数
        # 用于保存当前文本chunk
        def flush_text_chunk():

            # 使用外层变量
            nonlocal current_text

            # 使用外层变量
            nonlocal current_size

            # 如果当前没有文本
            if not current_text:
                # 直接返回
                return

            # 拼接文本
            text = "\n".join(current_text).strip()

            # 如果文本为空
            if not text:
                # 直接返回
                return

            # 对超长文本切分
            split_chunks = self.split_large_text(text)

            # 遍历切分结果
            for split_text in split_chunks:
                # 创建chunk并保存
                chunks.append(

                    self.create_chunk(

                        # chunk文本
                        text=split_text,

                        # 标题路径
                        title_path=title_stack,

                        # 内容类型
                        content_type="text",

                        # 来源文件
                        source=source,
                    )
                )

            # 清空当前文本
            current_text = []

            # 重置当前大小
            current_size = 0

        # 遍历所有token
        while i < len(tokens):

            # 当前token
            token = tokens[i]

            # ==================================
            # Heading
            # ==================================

            # 如果是标题开始
            if token.type == "heading_open":
                # flush之前正文
                flush_text_chunk()

                # 获取标题级别
                # h1 -> 1
                # h2 -> 2
                level = int(token.tag[1])

                # heading_open 后面通常是 inline token
                inline_token = tokens[i + 1]

                # 获取标题文本
                title = inline_token.content.strip()

                # 修正标题层级
                # 例如：
                # h1 -> h2 -> h3
                # 如果出现新h2
                # 需要截断旧h3
                title_stack[:] = title_stack[:level - 1]

                # 加入当前标题
                title_stack.append(title)

                # 跳过：
                # heading_open
                # inline
                # heading_close
                i += 3

                # 继续下一轮
                continue

            # ==================================
            # Fence Code Block
            # ==================================

            # 如果是代码块
            if token.type == "fence":

                # flush之前正文
                flush_text_chunk()

                # 获取代码文本
                code_text = token.content.strip()

                # 对超长代码切分
                code_chunks = self.split_large_text(code_text)

                # 遍历代码chunk
                for code in code_chunks:
                    # 创建代码chunk
                    chunks.append(

                        self.create_chunk(

                            # 代码文本
                            text=code,

                            # 标题路径
                            title_path=title_stack,

                            # 内容类型
                            content_type="code",

                            # 来源文件
                            source=source,
                        )
                    )

                # token索引+1
                i += 1

                # 继续下一轮
                continue

            # ==================================
            # Table
            # ==================================

            # 如果是table开始
            if token.type == "table_open":

                # flush正文
                flush_text_chunk()

                # 表格buffer
                table_buffer = []

                # 收集整个table内容
                while i < len(tokens):

                    # 当前token
                    t = tokens[i]

                    # 如果是inline文本
                    if t.type == "inline":
                        # 保存文本
                        table_buffer.append(t.content)

                    # 如果table结束
                    if t.type == "table_close":
                        # 跳出循环
                        break

                    # token索引+1
                    i += 1

                # 合并table文本
                table_text = "\n".join(table_buffer)

                # 创建table chunk
                chunks.append(

                    self.create_chunk(

                        # table文本
                        text=table_text,

                        # 标题路径
                        title_path=title_stack,

                        # 内容类型
                        content_type="table",

                        # 来源文件
                        source=source,
                    )
                )

                # token索引+1
                i += 1

                # 继续下一轮
                continue

            # ==================================
            # Paragraph / Inline
            # ==================================

            # 如果是inline文本
            if token.type == "inline":

                # 获取文本
                text = token.content.strip()

                # 如果文本不为空
                if text:

                    # 如果当前chunk放不下
                    if current_size + len(text) > self.max_chunk_size:
                        # flush旧chunk
                        flush_text_chunk()

                    # 加入当前文本buffer
                    current_text.append(text)

                    # 更新chunk大小
                    current_size += len(text)

            # token索引+1
            i += 1

        # 循环结束后flush最后chunk
        flush_text_chunk()

        # 返回chunks
        return chunks

    # 完整处理流程
    def process(

            self,

            # markdown文件路径
            file_path: str,
    ):

        # 读取markdown文件
        markdown_text = Path(file_path).read_text(

            # utf8编码
            encoding="utf-8"
        )

        # 解析markdown
        tokens = self.parse_markdown(markdown_text)

        # 构建chunks
        chunks = self.build_chunks(

            # token列表
            tokens=tokens,

            # 来源文件名
            source=Path(file_path).name,
        )

        # 返回chunks
        return chunks

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

    # 创建chunker实例
    chunker = MarkdownASTChunker(

        # 最大chunk长度
        max_chunk_size=1500,

        # overlap长度
        overlap=200,
    )

    # 处理markdown文件
    chunks = chunker.process("test.md")

    # 保存 json debug 文件
    chunker.save_chunks_to_json(
        chunks,
        "debug_chunks.json"
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

        # 拼接并输出标题路径
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

        # 只输出前1000字符
        print(chunk["text"][:1000])

        # 空行
        print("\n")
