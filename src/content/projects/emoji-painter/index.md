---
title: "🎨 Emoji Painter"
description: "A Transformer-based model that paints using emojis. You can provide images or emojis to paint with, and it will attempt to recreate images using emojis."
date: "05/15/2024"
demoURL: "https://replicate.com/johnsutor/emoji-painter"
repoURL: "https://github.com/johnsutor/emoji-painter"
---

## About 
This repo includes the code for teaching a model to paint using emojis. You can provide images or emojis to paint with, and it will attempt to recreate images using emojis.

Much of this code is adopted from the Paint Transformer paper. In this code base, I treat emojis like "brushes", and use a Gumbel Softmax-based lookup to choose an emoji to paste to the canvas during training (similar to how attention uses a softmax to select keys during training).