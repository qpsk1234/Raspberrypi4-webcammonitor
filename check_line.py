
filename = 'm:/vive-local/web_stream.py'
with open(filename, 'rb') as f:
    lines = f.readlines()

# Line 759 is index 758
line_757 = lines[756]
line_758 = lines[757]
line_759 = lines[758]

print(f"Line 757: {line_757}")
print(f"Line 758: {line_758}")
print(f"Line 759: {line_759}")

for i in [756, 757, 758, 759, 760]:
    if i < len(lines):
        print(f"Line {i+1}: {lines[i]}")
        for b in lines[i]:
            print(f"{b:02x}", end=' ')
        print()
