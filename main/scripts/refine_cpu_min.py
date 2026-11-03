with open('main/modes/cpu_min/summarizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_cond = 'if unwrapped and unwrapped[-1][-1] not in ".!?:;\\\"”":'
new_cond = 'if unwrapped and (unwrapped[-1][-1] not in ".!?:;\\\"”" or (unwrapped[-1].endswith(":") and not re.match(r"^[-*•\\d]", l))):'

if old_cond in content:
    content = content.replace(old_cond, new_cond, 1)

with open('main/modes/cpu_min/summarizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESSFULLY REFINED cpu_min/summarizer.py!')
