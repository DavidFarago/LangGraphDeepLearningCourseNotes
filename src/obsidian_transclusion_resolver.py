# see https://www.perplexity.ai/search/i-want-to-export-an-obsidian-n-ZOaOuDJLSEi2OgCizgNzZw
import re
import os

def convert_image_links(content):
    # Regex pattern for Obsidian image transclusions
    pattern = r'!\[\[([^|\]]+)(\|([^\]]+))?\]\]'
    
    def replacement(match):
        image_path = match.group(1)
        alt_or_width = match.group(3)
        
        # Modify the image path to include 'attachments/'
        new_image_path = f'attachments/{image_path}'
        
        if alt_or_width and alt_or_width.isdigit():
            return f'<img src="{new_image_path}" width="{alt_or_width}">'
        else:
            alt_text = alt_or_width or image_path.split('/')[-1]
            return f'![{alt_text}]({new_image_path})'

    return re.sub(pattern, replacement, content)

# Process all .md files in the parent directory
for filename in os.listdir('..'):
    if filename.endswith('.md'):
        with open(f'../{filename}', 'r', encoding='utf-8') as file:
            content = file.read()
        
        modified_content = convert_image_links(content)
        
        with open(f'../{filename}', 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        print(f"Processed: {filename}")

print("All .md files have been processed.")
