def print_tree(
    section,
    indent=0
):

    prefix = "  " * indent

    print(
        f"{prefix}"
        f"- {section.title} "
        f"(level={section.level})"
    )

    # =====================
    # blocks
    # =====================

    for block in section.blocks:

        preview = (
            block["content"]
            .replace("\n", " ")
            [:60]
        )

        print(
            f"{prefix}  "
            f"[{block['type']}] "
            f"{preview}"
        )

    # =====================
    # children
    # =====================

    for child in section.children:

        print_tree(
            child,
            indent + 1
        )