with open("main/modes/gpu_finetune/writer.py", "r", encoding="utf-8") as f:
    content = f.read()

target = "adapter=\"writer\",\n                )"
repl = "adapter=\"writer\",\n                    timeout=25.0,\n                )"

if target in content:
    content = content.replace(target, repl, 1)
    with open("main/modes/gpu_finetune/writer.py", "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated writer.py successfully")
else:
    print("target not found in writer.py")
