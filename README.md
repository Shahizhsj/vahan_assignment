# Research Paper Analysis Tool Documentation

## Overview
A comprehensive tool for analyzing research papers with features including file summarization, ArXiv paper search, URL summarization, DOI reference lookup, and cross-paper synthesis. The tool uses advanced AI models (Google's Gemini) for analysis and provides both text and audio outputs using ElevenLabs text-to-speech.

**Last Updated:** 2025-04-10
**Author:** Shahir

## Features

1. **File Summarizer**
   - Supports PDF, DOCX, TXT, CSV, and XLSX files
   - Generates concise summaries
   - Provides audio narration of summaries

2. **ArXiv Search**
   - Search academic papers on ArXiv
   - Sort by relevance, latest, or update date
   - Get paper summaries with audio narration

3. **URL Summarizer**
   - Extract and summarize content from web pages
   - Generate audio summaries

4. **DOI Reference**
   - Look up papers using DOI
   - Get detailed paper information and summaries

5. **Cross-paper Synthesis**
   - Classify papers into user-defined topics
   - Generate synthesized summaries across papers

# Samples output

## I uploaded a paper on Inception model and you can see the summary and podcast of that summary

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/8ef22471733e8627a30ae9787943ddb3dbd19cdd/Screenshot%20(199).png)


## I inserted a link of a paper form Arvix and you can see the summary of that paper and related podcast of that paper

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/8ef22471733e8627a30ae9787943ddb3dbd19cdd/Screenshot%20(200).png)


## Here I gave the input as DOI Reference number and we get summarized output along with the podcast of that paper

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/8ef22471733e8627a30ae9787943ddb3dbd19cdd/Screenshot%20(201).png)

## Here I give the topics as NLP and Computer vision and we get the classification of papers accorading to research paper and we also get the topic wise synthesis

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/8ef22471733e8627a30ae9787943ddb3dbd19cdd/Screenshot%20(202).png)

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/8ef22471733e8627a30ae9787943ddb3dbd19cdd/Screenshot%20(203).png)


# System architecture

![File Summarization](https://github.com/Shahizhsj/vahan_assignment/blob/48951cddcf39f3d3acc2bcd4dafb51b421f6b7e1/workflow.png)

# Setup instructions

### Prerequisites
- **Python:** Version 3.9 or higher.
- **Internet Connection:** Required for API calls.
- **API Keys:**  
  - **Google Gemini API Key** for text processing.
  - **ElevenLabs API Key** for text-to-speech.
### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shahizhsj/research-paper-analysis.git
   cd research-paper-analysis
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the Application**
   ```bash
   python app.py
   ```

6. **Access the Interface**
   Open your browser and go to:  
   `http://localhost:7860` (default port).

# Multi-agent design and coordination approach

## Agent Types and Responsibilities

## 1. Document Processing Agent
- **What it does:** Extracts text from your files, URLs, or research IDs (like DOIs).  
- **Job:** Identifies file types (PDF, webpage), pulls out text, and cleans it up (no odd characters).  
- **Example:** You upload a PDF; it grabs the text and splits it into chunks.

## 2. Analysis Agent
- **What it does:** Finds key info in the text and summarizes it.  
- **Job:** Creates searchable text chunks with Google Gemini, stores them in FAISS, and writes quick summaries.  
- **Example:** You ask about a paper; it fetches the main points and sums them up.

## 3. Synthesis Agent
- **What it does:** Groups and connects everything by topic.  
- **Job:** Sorts summaries into your topics (e.g., “AI Ethics”) and weaves them into a clear overview.  
- **Example:** You pick “AI stuff”; it combines related papers into one story.

# Audio generation implementation
For audio generation, I used the ElevenLabs API.  
I chose the `eleven_flash_v2` model from ElevenLabs because it’s very fast.

# Limitations
1. Our code relies on APIs, which have limited usage time.
2. Generating large audio files causes latency.
3. Complex scientific diagrams are not processed.

# Future Improvements
1. Add support for mathematical notations and diagrams.
2. Replace APIs with pretrained models for greater flexibility.
3. Use more advanced reasoning models, such as those from OpenAI.
