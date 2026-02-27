
import re

filename = 'm:/vive-local/web_stream.py'
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start and end of the script in TEMPLATE
# We look for <script> and </script> inside the TEMPLATE = """ block
template_match = re.search(r'TEMPLATE = """(.*?)"""', content, re.DOTALL)
if template_match:
    template_str = template_match.group(1)
    script_match = re.search(r'<script>(.*?)</script>', template_str, re.DOTALL)
    if script_match:
        script_code = script_match.group(1)
        
        # Check braces
        stack = []
        for i, char in enumerate(script_code):
            if char == '{':
                stack.append(('{', i))
            elif char == '}':
                if not stack:
                    print(f"Excess closing brace }} at position {i}")
                else:
                    stack.pop()
        
        for char, pos in stack:
            print(f"Unclosed open brace {char} at position {pos}")
            
        if not stack:
            print("Braces are balanced.")
        
        # Check quotes
        # Simple check for balanced quotes on each line if not inside backticks
        lines = script_code.splitlines()
        for i, line in enumerate(lines):
            # This is very naive but might find obvious mistakes
            # Skip lines with template literals for now
            if '`' not in line:
                dquotes = line.count('"')
                squotes = line.count("'")
                if dquotes % 2 != 0:
                    print(f"Line {i+1}: Unbalanced double quotes: {line.strip()}")
                if squotes % 2 != 0:
                    print(f"Line {i+1}: Unbalanced single quotes: {line.strip()}")
    else:
        print("Script not found in TEMPLATE.")
else:
    print("TEMPLATE not found.")
