"""
代码文件合并工具.

该工具将多个代码文件合并为单个文本报告，便于代码审查、文档生成或提交给AI助手分析。
支持直接传入文件路径或从列表文件批量读取，自动过滤二进制文件和无效路径。

功能特性：
    1. 智能文件过滤：自动检测并跳过二进制文件、文件夹、不存在路径
    2. 双模式输入：支持命令行直接传参或从文本列表文件读取
    3. 结构化输出：带编号的文件分隔、清晰的头部统计信息
    4. 错误处理：单个文件读取失败不影响整体流程
    5. 路径安全：自动解析为绝对路径，避免相对路径混淆

输出格式：
    代码合并报告
    包含文件数量: N
    ================================================================================
    
    ### 文件 1/N: /path/to/file1.py ###
    ----------------------------------------
    [文件1内容]
    
    ================================================================================
    
    ### 文件 2/N: /path/to/file2.py ###
    ----------------------------------------
    [文件2内容]

使用方法：
    # 方式1：直接传参
    python -m source.tools.code_summary ./src/main.py ./utils/helper.py ./config.yaml
    
    # 方式2：从列表文件读取（支持注释行#）
    python -m source.tools.code_summary --list ./source/tools/path.txt
    
    # file_list.txt 示例：
    # 这是注释行，会被忽略
    ./src/main.py
    ./utils/helper.py
    ./README.md
"""

import sys
import os
from pathlib import Path

# ====================== 配置 ======================

# 获取当前脚本文件的绝对路径，并取其父目录作为基准路径
SCRIPT_DIR = Path(__file__).resolve().parent
# 输出文件路径：脚本同级目录下的 code_summary.txt
OUTPUT_FILENAME = SCRIPT_DIR / "code_summary.txt"

# 文件分隔线样式（80个等号）
SEPARATOR_LINE = "=" * 80


# ====================== 内部辅助函数 ======================

def is_binary_file(file_path: Path) -> bool:
    """
    检测文件是否为二进制文件.

    算法：读取文件前1024字节，检查是否包含空字节（\x00）。
    文本文件通常不包含空字节，而二进制文件（图片、可执行文件等）通常包含。

    Args:
        file_path: 待检测的文件路径。

    Returns:
        bool: True 如果检测到空字节（判定为二进制），False 否则。
            若读取失败也返回True（保守策略，避免处理损坏文件）。

    Note:
        这是启发式检测，对于某些特殊编码文本可能误判，
        但对于常见代码文件（UTF-8、ASCII）足够可靠。
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        # 读取失败时保守地认为是二进制（或损坏），跳过处理
        return True


def process_files(file_paths: list, output_file: Path):
    """
    处理文件列表，合并写入到单个输出文件.

    处理流程：
        1. 路径验证：检查存在性、文件类型、二进制检测
        2. 有效文件统计与报告
        3. 顺序读取并写入，添加结构化分隔标记
        4. 错误隔离：单个文件失败不影响其他文件

    Args:
        file_paths: 输入文件路径字符串列表。
        output_file: 输出文件的Path对象。

    Returns:
        None. 结果直接写入磁盘文件。

    Side Effects:
        创建/覆盖 output_file 指定的文件。
        打印处理进度到标准输出。

    Examples:
        >>> paths = ["./test.py", "./README.md"]
        >>> process_files(paths, Path("./output.txt"))
        🔍 正在检查 2 个文件路径...
        ✅ 发现 2 个有效文件，正在生成 output.txt...
        🎉 完成！所有代码已保存至: /absolute/path/to/output.txt
    """
    valid_files = []
    
    print(f"🔍 正在检查 {len(file_paths)} 个文件路径...")
    
    # ----- 第一阶段：路径验证与过滤 -----
    for path_str in file_paths:
        # 清理输入：去除首尾空白字符（处理复制粘贴带来的空格）
        path = Path(path_str.strip())
        
        # 1. 检查路径是否存在（文件或文件夹）
        if not path.exists():
            print(f"   ⚠️ 跳过 (不存在): {path}")
            continue
            
        # 2. 检查是否是文件（排除文件夹）
        if path.is_dir():
            print(f"   ⚠️ 跳过 (是文件夹): {path}")
            continue
            
        # 3. 检查是否为二进制文件（避免将图片等混入文本报告）
        if is_binary_file(path):
            print(f"   ⚠️ 跳过 (二进制文件): {path}")
            continue
            
        valid_files.append(path)

    # 无有效文件时提前退出
    if not valid_files:
        print("❌ 没有找到有效的代码文件。")
        return

    # ----- 第二阶段：合并写入 -----
    print(f"✅ 发现 {len(valid_files)} 个有效文件，正在生成 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入报告头部信息
        outfile.write(f"代码合并报告\n")
        outfile.write(f"包含文件数量: {len(valid_files)}\n")
        outfile.write(f"{SEPARATOR_LINE}\n\n")

        # 逐个处理有效文件
        for i, file_path in enumerate(valid_files, 1):
            # 写入文件标记头（包含序号和绝对路径）
            outfile.write(f"### 文件 {i}/{len(valid_files)}: {file_path} ###\n")
            outfile.write("-" * 40 + "\n")
            
            try:
                # 读取文件内容（忽略编码错误，保证流程不中断）
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                # 单个文件错误不影响整体，记录错误信息后继续
                outfile.write(f"[读取错误: {e}]")
            
            # 写入分隔符（空行 + 分隔线 + 空行）
            outfile.write(f"\n\n{SEPARATOR_LINE}\n\n")

    # 提示用户文件的绝对路径，便于查找
    print(f"🎉 完成！所有代码已保存至: {output_file.resolve()}")


# ====================== 主入口 ======================

def main():
    """
    命令行入口函数，处理参数解析和流程控制.

    支持的命令行模式：
        1. 直接传参模式：python code_summary.py <file1> <file2> ...
        2. 列表文件模式：python code_summary.py --list <list_file>

    列表文件格式：
        - 每行一个文件路径
        - 以 # 开头的行为注释，会被忽略
        - 空行自动跳过

    Args:
        通过 sys.argv 接收命令行参数。

    Returns:
        None. 根据参数调用 process_files 或直接打印帮助信息。

    Examples:
        $ python code_summary.py ./src/main.py ./utils.py
        $ python code_summary.py --list files.txt
        $ python code_summary.py
        用法说明:
          1. 直接传入文件路径: python code_summary.py ./bin/1.py ./src/main.py
          2. 从文件读取列表:   python code_summary.py --list paths.txt
    """
    # 获取命令行参数（排除脚本名本身）
    args = sys.argv[1:]
    
    file_list = []

    # 无参数时显示帮助信息
    if not args:
        print("用法说明:")
        print("  1. 直接传入文件路径: python code_summary.py ./bin/1.py ./src/main.py")
        print("  2. 从文件读取列表:   python code_summary.py --list paths.txt")
        return

    # ----- 模式1：从列表文件读取 -----
    if args[0] == "--list":
        if len(args) < 2:
            print("错误: 请指定包含路径列表的文件名")
            return
        
        list_file = Path(args[1])
        
        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                # 过滤：非空行且非注释行（以#开头）
                file_list = [
                    line.strip() 
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
        else:
            print(f"错误: 列表文件 {list_file} 不存在")
            return
    
    # ----- 模式2：直接传参 -----
    else:
        file_list = args

    # 执行核心处理逻辑
    process_files(file_list, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()