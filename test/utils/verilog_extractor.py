import re

def extract_verilog_code(file_content, remove_head=False, add_backtick=False):
    """
    从文本中提取删除注释后的verilog代码，一个用途是从LLM的生成中提取代码。
    """
    # 推理模型
    pattern = r"<answer>(.*?)</answer>"
    if re.findall(pattern, file_content, re.DOTALL):
        file_content = re.findall(pattern, file_content, re.DOTALL)[-1]
    # 去除注释
    note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
    file_content = re.sub(note_pattern, "", file_content)

    # 匹配 module 到 endmodule 之间的内容
    pattern1 = r"```verilog(.*?)```"
    result = re.findall(pattern1, file_content, re.DOTALL)
    result = result[-1] if len(result) else ""
    if not result:
        pattern2 = r"module\s+\w+\s*\([^)]*\)\s*;(?:[^m]|m(?!odule))*endmodule"
        result = re.findall(pattern2, file_content)
        result = result[-1] if len(result) else ""
    result = result.replace(";", ";\n").replace(";\n\n", ";\n")

    if remove_head:
        # Verilog eval v1, 匹配 module 头部
        pattern = r"module\s+\w+\s*\([^)]*\)\s*;(?:[^m]|m(?!odule))*endmodule"
        modules = re.findall(pattern, result)
        # 考虑多个module的情况，把top_module挪到最上面再删掉head
        if len(modules) >= 2 and modules[0].find("top_module") == -1 and modules[0].find("TopModule") == -1:
            for i, module in enumerate(modules):
                if module.find("top_module") != -1 or module.find("TopModule") != -1:
                    modules = [module] + modules[:i] + modules[i + 1:]
                    break
            result = '\n\n'.join(modules)
        head_pattern = r"module\s+\w+\s*\([^)]*\)\s*;"
        result = re.sub(head_pattern, "", result, count=1)

    if add_backtick:
        result = "```verilog\n" + result + "\n```"
    
    return result