
import sys

filename = 'm:/vive-local/web_stream.py'
try:
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Check for byte sequence that might be invalid in UTF-8
    try:
        data.decode('utf-8')
        print("File is valid UTF-8.")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        sys.exit(1)

    # Check for common problematic characters
    content = data.decode('utf-8')
    lines = content.splitlines()
    for i, line in enumerate(lines):
        # Browser line 719 might be around here.
        if 500 < i < 850: 
            for j, char in enumerate(line):
                if ord(char) > 127:
                    print(f"Non-ASCII character {char} (U+{ord(char):04X}) at line {i+1}, col {j+1}")
                if ord(char) < 32 and char not in '\n\r\t':
                    print(f"Control character (U+{ord(char):04X}) at line {i+1}, col {j+1}")

except Exception as e:
    print(f"Error: {e}")
