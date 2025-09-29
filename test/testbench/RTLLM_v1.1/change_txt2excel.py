import json
import openpyxl

def read_sixth_line_as_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        result_dic_line_index = -1
        for i, line in enumerate(lines):
            if 'result_dic' in line:
                result_dic_line_index = i + 1
                break
        
        if result_dic_line_index == -1 or result_dic_line_index >= len(lines):
            raise ValueError("未找到result_dic或其后没有内容")
        
        target_line = lines[result_dic_line_index].strip()
        print(f"原始目标行内容: {target_line}")  # 调试信息
        
        # 将单引号替换为双引号
        target_line_fixed = target_line.replace("'", '"')
        # print(f"修正后的目标行内容: {target_line_fixed}")  # 调试信息
        
        try:
            data_dict = json.loads(target_line_fixed)
        except json.JSONDecodeError as e:
            raise ValueError(f"目标行不是有效的JSON格式: {e}")
    return data_dict

def write_to_excel(data_dict, excel_file_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    
    keys = list(data_dict.keys())
    syntax_success_values = [data_dict[key]['syntax_success'] for key in keys]
    func_success_values = [data_dict[key]['func_success'] for key in keys]
    
    ws.append(keys)
    ws.append(syntax_success_values)
    ws.append(func_success_values)
    
    wb.save(excel_file_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str) # 文件的路径
    args = parser.parse_args()
    txt_file_path = args.path # 对应的txt文件的路径
    excel_file_path = txt_file_path[:-len(".txt")]+'.xlsx'  # 替换为你想要保存的Excel文件路径
    
    try:
        data_dict = read_sixth_line_as_dict(txt_file_path)
        write_to_excel(data_dict, excel_file_path)
        print(f"数据已成功写入 {excel_file_path}")
    except Exception as e:
        print(f"发生错误: {e}")



