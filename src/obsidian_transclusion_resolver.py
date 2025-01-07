# see https://www.perplexity.ai/search/i-want-to-export-an-obsidian-n-ZOaOuDJLSEi2OgCizgNzZw
import re
import os

def convert_links(content):
    # Regex pattern for Obsidian image links
    image_pattern = r'!\[\[([^|\]]+)(\|([^\]]+))?\]\]'
    
    # Regex pattern for specific file links
    file_pattern = r'\[\[([^|\]]+\.py)\|([^|\]]+\.py)\]\]'

    def image_replacement(match):
        image_path = match.group(1)
        alt_or_width = match.group(3)
        
        new_image_path = f'attachments/{image_path}'
        
        if alt_or_width and alt_or_width.isdigit():
            return f'<img src="{new_image_path}" width="{alt_or_width}">'
        else:
            alt_text = alt_or_width or image_path.split('/')[-1]
            return f'![{alt_text}]({new_image_path})'

    def file_replacement(match):
        file_name = match.group(1)
        return f'[{file_name}](src/{file_name})'

    content = re.sub(image_pattern, image_replacement, content)
    content = re.sub(file_pattern, file_replacement, content)
    
    return content

# Process all .md files in the parent directory
for filename in os.listdir('..'):
    if filename.endswith('.md'):
        with open(f'../{filename}', 'r', encoding='utf-8') as file:
            content = file.read()
        
        modified_content = convert_links(content)
        
        with open(f'../{filename}', 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        print(f"Processed: {filename}")

print("All .md files have been processed.")
