from markdown_it import MarkdownIt


class MarkdownSection:

    def __init__(
        self,
        level,
        title,
        parent_path=None
    ):

        self.level = level

        self.title = title

        self.parent_path = parent_path or []

        self.blocks = []

        self.children = []

    @property
    def path(self):

        return self.parent_path + [self.title]


class MarkdownASTParser:

    def __init__(self):

        self.md = MarkdownIt()

    def parse(self, markdown_text):

        tokens = self.md.parse(markdown_text)

        root = MarkdownSection(
            level=0,
            title="ROOT"
        )

        stack = [root]

        i = 0

        while i < len(tokens):

            token = tokens[i]

            # ---------------------------------
            # heading
            # ---------------------------------

            if token.type == "heading_open":

                level = int(token.tag[1])

                title = tokens[i + 1].content.strip()

                while (
                    stack
                    and stack[-1].level >= level
                ):
                    stack.pop()

                parent = stack[-1]

                section = MarkdownSection(
                    level=level,
                    title=title,
                    parent_path=parent.path
                    if parent.title != "ROOT"
                    else []
                )

                parent.children.append(section)

                stack.append(section)

                i += 3
                continue

            # ---------------------------------
            # code block
            # ---------------------------------

            current = stack[-1]

            if token.type in [
                "fence",
                "code_block"
            ]:

                current.blocks.append({
                    "type": "code",
                    "content": token.content
                })

            # ---------------------------------
            # table
            # ---------------------------------

            elif token.type == "table_open":

                table_content = []

                j = i

                while (
                    j < len(tokens)
                    and tokens[j].type != "table_close"
                ):

                    if tokens[j].content.strip():
                        table_content.append(
                            tokens[j].content
                        )

                    j += 1

                current.blocks.append({
                    "type": "table",
                    "content": "\n".join(table_content)
                })

            # ---------------------------------
            # paragraph
            # ---------------------------------

            elif token.type == "inline":

                if token.content.strip():

                    current.blocks.append({
                        "type": "paragraph",
                        "content": token.content
                    })

            i += 1

        return root