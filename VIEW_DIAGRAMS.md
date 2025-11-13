# How to View the Sequence Diagrams

There are several ways to visualize the Mermaid sequence diagrams in `SEQUENCE_DIAGRAMS.md`:

## Option 1: Open the HTML Viewer (Easiest) ⭐

I've created a standalone HTML file that renders all diagrams:

```bash
# Simply open the HTML file in your browser
open view_diagrams.html
```

Or double-click `view_diagrams.html` in your file explorer. This works offline and doesn't require any setup!

## Option 2: VS Code with Mermaid Extension

1. Install the **"Markdown Preview Mermaid Support"** extension in VS Code:
   - Open VS Code
   - Go to Extensions (Cmd+Shift+X on Mac, Ctrl+Shift+X on Windows/Linux)
   - Search for "Markdown Preview Mermaid Support"
   - Install it

2. Open `SEQUENCE_DIAGRAMS.md` in VS Code

3. Press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows/Linux) to open the preview

4. The diagrams will render automatically!

## Option 3: Online Mermaid Live Editor

1. Go to [https://mermaid.live](https://mermaid.live)

2. Copy a diagram code block from `SEQUENCE_DIAGRAMS.md` (the content between ` ```mermaid` and ` ``` `)

3. Paste it into the editor

4. The diagram will render instantly!

## Option 4: GitHub/GitLab

If you push the repository to GitHub or GitLab, the diagrams will automatically render in the markdown file when viewed on the platform.

## Option 5: Python Script to Generate Images

You can also generate PNG/SVG images from the diagrams:

```bash
# Install mermaid-cli (requires Node.js)
npm install -g @mermaid-js/mermaid-cli

# Generate images from the markdown file
mmdc -i SEQUENCE_DIAGRAMS.md -o diagrams_output/
```

## Option 6: Use a Markdown Viewer App

Many markdown viewers support Mermaid:
- **Typora** (paid, but excellent)
- **Obsidian** (free, great for notes)
- **Mark Text** (free, open-source)

## Recommended Approach

For the quickest viewing experience, use **Option 1** (the HTML file) - just open `view_diagrams.html` in your browser!

For editing and viewing together, use **Option 2** (VS Code with extension).

