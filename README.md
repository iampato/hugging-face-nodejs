````markdown
# Hugging Face Node.js

A Node.js project for generating **text and image embeddings** and transforming user profiles into engaging dating bios using Hugging Face models.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Build User Profile Text](#build-user-profile-text)  
  - [Generate Text Prompt](#generate-text-prompt)  
  - [Text Embedding](#text-embedding)  
  - [Image Embedding](#image-embedding)  
- [Supported Models](#supported-models)  
- [Performance Notes](#performance-notes)  
- [License](#license)  

---

## Overview

This project demonstrates how to use Hugging Face Transformers in Node.js for:

1. **Text generation** – rewrite user bios into funny, engaging dating bios.  
2. **Text embeddings** – extract numerical embeddings for text using sentence-transformers models.  
3. **Image embeddings** – extract visual embeddings from images using CLIP models.  

The project is designed to support **scalable embedding pipelines** and can be used in applications like dating apps, recommendation engines, or AI-powered content analysis.

---

## Features

- Convert raw user bio and interests into a clean profile text.  
- Generate enhanced, engaging dating bios using Hugging Face text-generation models.  
- Extract **text embeddings** for semantic similarity or vector search.  
- Extract **image embeddings** using pre-trained CLIP models.  
- Logs timing for pipeline loading and embedding generation.  

---

## Installation

```bash
# Clone the repo
git clone https://github.com/iampato/hugging-face-nodejs.git
cd hugging-face-nodejs

# Install dependencies
npm install
# or
yarn install
````

Make sure you are using Node.js 18+ for ES module support.

---

## Usage

### Build User Profile Text

```js
import { buildUserProfileText } from './index.js';

const user = {
  bio: "Curious traveler who loves coffee and photography",
  interests: ["AI", "hiking", "books"],
};

const profileText = buildUserProfileText(user);
console.log(profileText);
// Output: "User dating profile, Bio:Curious traveler who loves coffee and photography. Their interests include AI, hiking, books."
```

---

### Generate Text Prompt

```js
import { generatePrompt } from './index.js';

const promptText = buildUserProfileText(user);

const generatedPromptText = await generatePrompt({
    modelId: "HuggingFaceTB/SmolLM2-360M-Instruct",
    inputText: promptText,
    maxLength: 400
});

console.log(generatedPromptText);
```

This generates a polished, engaging dating bio from the input text.

---

### Text Embedding

```js
import { getTextEmbedding, extractEmbedding } from './index.js';

const embeddingResult = await getTextEmbedding({
    modelId: "sentence-transformers/all-MiniLM-L6-v2",
    params: promptText
});

const embeddingArray = extractEmbedding(embeddingResult);
console.log(embeddingArray.length); // 384 for MiniLM-L6
```

---

### Image Embedding

```js
import { getImageEmbedding, extractEmbedding } from './index.js';

const imageEmbeddingResult = await getImageEmbedding({
    modelId: "Xenova/clip-vit-base-patch32",
    imageUrl: "https://example.com/image.jpg"
});

const imageEmbeddingArray = extractEmbedding(imageEmbeddingResult);
console.log(imageEmbeddingArray.length); // depends on the model
```

---

## Supported Models

### Text Embeddings

* `sentence-transformers/all-MiniLM-L6-v2` (384D, fast)
* `sentence-transformers/all-mpnet-base-v2` (768D, better quality)
* `BAAI/bge-small-en-v1.5` (384D, good performance)
* `BAAI/bge-base-en-v1.5` (768D, high quality)

### Image Embeddings

* `Xenova/clip-vit-base-patch32`
* Other CLIP-based vision models on Hugging Face

### Text Generation

* `HuggingFaceTB/SmolLM-135M-Instruct`
* `HuggingFaceTB/SmolLM2-360M-Instruct`
* `Xenova/gpt2` (small-scale testing)

---

## Performance Notes

* **Pipeline load** is slow the first time due to model download & initialization. Subsequent runs are faster thanks to caching.
* For batch processing, reuse the **pipeline objects** to avoid repeated loading.
* Timing logs are provided using `console.time()` / `console.timeEnd()` for profiling.

---

## License

MIT License.
Feel free to use and modify for personal or commercial projects.

