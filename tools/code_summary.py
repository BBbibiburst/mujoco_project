import sys
import os
from pathlib import Path

# --- 配置 ---
# 修改点：获取当前脚本文件 (__file__) 的绝对路径，并取其父目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 将输出文件路径设置为脚本同级目录下的 code_summary.txt
OUTPUT_FILENAME = SCRIPT_DIR / "code_summary.txt"

SEPARATOR_LINE = "=" * 80

def is_binary_file(file_path: Path) -> bool:
    """
    简单的二进制文件检测
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True

def process_files(file_paths: list, output_file: Path):
    """
    接收文件路径列表，合并写入到一个文件
    """
    valid_files = []
    
    print(f"🔍 正在检查 {len(file_paths)} 个文件路径...")
    
    for path_str in file_paths:
        # 修复点：先 strip 字符串，再转为 Path
        path = Path(path_str.strip())
        
        # 1. 检查路径是否存在
        if not path.exists():
            print(f"   ⚠️ 跳过 (不存在): {path}")
            continue
            
        # 2. 检查是否是文件 (不是文件夹)
        if path.is_dir():
            print(f"   ⚠️ 跳过 (是文件夹): {path}")
            continue
            
        # 3. 检查是否为二进制文件
        if is_binary_file(path):
            print(f"   ⚠️ 跳过 (二进制文件): {path}")
            continue
            
        valid_files.append(path)

    if not valid_files:
        print("❌ 没有找到有效的代码文件。")
        return

    # 修改点：输出时使用 output_file 的绝对路径字符串，方便用户查看
    print(f"✅ 发现 {len(valid_files)} 个有效文件，正在生成 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入头部
        outfile.write(f"代码合并报告\n")
        outfile.write(f"包含文件数量: {len(valid_files)}\n")
        outfile.write(f"{SEPARATOR_LINE}\n\n")

        for i, file_path in enumerate(valid_files, 1):
            # 写入文件标记
            outfile.write(f"### 文件 {i}/{len(valid_files)}: {file_path} ###\n")
            outfile.write("-" * 40 + "\n")
            
            try:
                # 读取内容
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"[读取错误: {e}]")
            
            # 写入分隔符
            outfile.write(f"\n\n{SEPARATOR_LINE}\n\n")

    # 提示用户文件生成的具体绝对路径
    print(f"🎉 完成！所有代码已保存至: {output_file.resolve()}")

def main():
    args = sys.argv[1:]
    
    file_list = []

    if not args:
        print("用法说明:")
        print("  1. 直接传入文件路径: python code_summary.py ./bin/1.py ./src/main.py")
        print("  2. 从文件读取列表:   python code_summary.py --list paths.txt")
        return

    if args[0] == "--list":
        if len(args) < 2:
            print("错误: 请指定包含路径列表的文件名")
            return
        list_file = Path(args[1])
        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                file_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            print(f"错误: 列表文件 {list_file} 不存在")
            return
    else:
        file_list = args

    process_files(file_list, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()