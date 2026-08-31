import re

def md_to_html(md_text):
    lines = md_text.split('\n')
    html_lines = []
    in_table = False
    table_header_done = False
    
    for line in lines:
        stripped = line.strip()
        
        # Headers
        if stripped.startswith('# '):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            html_lines.append(f'<h1>{stripped[2:]}</h1>')
            continue
        elif stripped.startswith('## '):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            html_lines.append(f'<h2>{stripped[3:]}</h2>')
            continue
        elif stripped.startswith('### '):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            html_lines.append(f'<h3>{stripped[4:]}</h3>')
            continue
            
        # Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped[1:-1].split('|')]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                table_header_done = True
                continue
            
            if not in_table:
                in_table = True
                table_header_done = False
                html_lines.append('<table><thead><tr>')
                for c in cells:
                    html_lines.append(f'<th>{format_inline(c)}</th>')
                html_lines.append('</tr></thead><tbody>')
            else:
                html_lines.append('<tr>')
                for c in cells:
                    html_lines.append(f'<td>{format_inline(c)}</td>')
                html_lines.append('</tr>')
            continue
        else:
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
                
        # Empty lines
        if not stripped:
            continue
            
        # Lists
        if stripped.startswith('- ') or stripped.startswith('• '):
            html_lines.append(f'<ul><li>{format_inline(stripped[2:])}</li></ul>')
            continue
            
        # Paragraphs
        html_lines.append(f'<p>{format_inline(line)}</p>')
        
    if in_table:
        html_lines.append('</tbody></table>')
        
    return '\n'.join(html_lines)

def format_inline(text):
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    return text

html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {
    size: letter;
    margin: 0.8in;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.0;
    color: #111;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 13pt;
    font-weight: 700;
    margin-top: 0pt;
    margin-bottom: 3pt;
    color: #0b2545;
    line-height: 1.1;
}
h2 {
    font-size: 11pt;
    font-weight: 700;
    margin-top: 6pt;
    margin-bottom: 3pt;
    border-bottom: 1px solid #ddd;
    padding-bottom: 1px;
    page-break-after: avoid;
    color: #133c55;
    line-height: 1.1;
}
h3 {
    font-size: 10pt;
    font-weight: 600;
    margin-top: 5pt;
    margin-bottom: 2pt;
    page-break-after: avoid;
    color: #205072;
    line-height: 1.1;
}
p {
    margin-top: 0pt;
    margin-bottom: 3pt;
    text-align: justify;
    line-height: 1.0;
}
ul {
    margin-top: 0pt;
    margin-bottom: 4pt;
    margin-left: 16px;
    padding: 0;
}
li {
    margin-bottom: 2pt;
    line-height: 1.0;
}
table {
    border-collapse: collapse;
    width: 100%;
    page-break-inside: avoid;
    margin-top: 4pt;
    margin-bottom: 4pt;
    font-size: 9.5pt;
    line-height: 1.0;
}
th, td {
    border: 1px solid #bbb;
    padding: 2.5pt 4.5pt;
    text-align: left;
    font-size: 9.5pt;
    line-height: 1.0;
}
th {
    background-color: #f2f4f7;
    font-weight: 600;
}
code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f5;
    padding: 1px 2px;
    border-radius: 2px;
}
img {
    max-width: 240pt;
    width: 240pt;
    height: auto;
    display: block;
    margin-top: 2pt;
    margin-bottom: 2pt;
}
</style>
</head>
<body>
__CONTENT__
</body>
</html>
"""

with open('PROPOSAL.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

body_html = md_to_html(md_text)
full_html = html_template.replace('__CONTENT__', body_html)

with open('PROPOSAL.html', 'w', encoding='utf-8') as f:
    f.write(full_html)
print("HTML generated successfully: PROPOSAL.html")
