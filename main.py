import requests
import io
from pydub import AudioSegment
from IPython.display import Audio
import simpleaudio as sa
import os
from langchain_core.prompts import PromptTemplate
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import UnstructuredFileLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain_community.tools import E2BDataAnalysisTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyC6TmVb5Vk5J0r6z0oCmjvNgzbblDKuf3Y")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyC6TmVb5Vk5J0r6z0oCmjvNgzbblDKuf3Y",
    # other params...
)
#Used for playing the audio file in gradio app
def elevenlabs_text_to_speech_and(text):
    voice_id = "EXAVITQu4vr4xnSDxMaL"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": "sk_5aab40b4a9867c89c184a141277edf83edc3bbfb82e93d70",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.7
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        audio_data = response.content
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_data = io.BytesIO()
        audio_segment.export(wav_data, format="wav")
        display(Audio(wav_data.getvalue(), autoplay=True))  # Plays automatically
    else:
        print("Error:", response.status_code, response.text)
#Used for generating the audio file in google cloab as sometimes audio may not play in gradio
def elevenlabs_text_to_speech_and_play(text):
    voice_id = "EXAVITQu4vr4xnSDxMaL"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": "sk_5aab40b4a9867c89c184a141277edf83edc3bbfb82e93d70",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.7
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        # Return the audio data directly
        return response.content
    else:
        print("Error:", response.status_code, response.text)
        return None
#prompt for LLM to summarize the research papers
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
#this prompt is used for the cross synthesis summarization
cross_systhesis_prompt="""you are an expert at anaylizing the research papers.your task is to classify research papers summaries in to one of the user given topic list and after that 
you need to synthesis the papers in each topic .you should summaries for every topic according to the research papers that you classified in that topic
"""
#From Here on we will be building the interface for easy access 

topic=[]
qa = None
image = None
t = 0
sumaries=[]

def vector(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyC6TmVb5Vk5J0r6z0oCmjvNgzbblDKuf3Y")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    })
    return qa

def file_upload(file):
    global sumaries
    global qa  # Indicate that we're modifying the global qa variable
    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        qa = vector(documents)
        answer = qa.run({"query": 'give a title and summarize the doucuemnt in 5 to 6 lines'})
        sumaries.append(answer)
        audio = elevenlabs_text_to_speech_and_play(answer)
        return answer,audio
    elif file.name.endswith('.docx'):
        loader = UnstructuredFileLoader(file)
        documents = loader.load()
        qa = vector(documents)
        answer = qa.run({"query": 'give a title and summarize the doucuemnt in 5 to 6 lines'})
        sumaries.append(answer)
        audio = elevenlabs_text_to_speech_and_play(answer)
        return answer,audio
    elif file.name.endswith('.txt'):
        loader = UnstructuredFileLoader(file)
        documents = loader.load()
        qa = vector(documents)
        answer = qa.run({"query": 'summarize the doucuemnt in 5 to 6 lines'})
        sumaries.append(answer)
        elevenlabs_text_to_speech_and_play(answer)
        return answer
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file.name)

        qa = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        answer = qa.run({"query": 'summarize the doucuemnt in 5 to 6 lines'})
        sumaries.append(answer)
        elevenlabs_text_to_speech_and_play(answer)
        return answer
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file.name)
        qa = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        answer = qa.run({"query": 'summarize the doucuemnt in 7 to 8 lines'})
        sumaries.append(answer)
        elevenlabs_text_to_speech_and_play(answer)
        return answer
    else:
      pass
def url(link):
    try:
        loader = WebBaseLoader(link)
        docs = loader.load()
        if not docs:
            return "Error: No content could be extracted from the URL"
        qa = vector(docs)
        answer = qa.run({"query": 'summarize the research paper in 7 to 8 lines'})
        sumaries.append(answer)
        audio = elevenlabs_text_to_speech_and_play(answer)
        return answer, audio
    except Exception as e:
        return f"Error processing URL: {str(e)}"
def doi(doi_number):
  client = arxiv.Client()
  arxiv_search = arxiv.Search(id_list=[str(doi_number)])
  formatted_result=""
  for i, paper in enumerate(client.results(arxiv_search), 1):
    formatted_result += f"""
Paper {i}
{'='*50}
Title: {paper.title}
Authors: {format_authors(paper.authors)}

Summary:
{paper.summary}

Link: {paper.entry_id}
{'='*50}

"""
    if formatted_result:
        sumaries.append(formatted_result.strip())
        audio = elevenlabs_text_to_speech_and_play(formatted_result.strip())
        return formatted_result.strip(),audio
    else:
        return "No results found!"


def topic_adder(t):
  global topic
  topic.append(t)
  return topic
def format_authors(authors):
    if isinstance(authors, list):
        return ", ".join([str(author) for author in authors])
    return str(authors)
def cross_ana():
  global topic
  global sumaries
  new_sum='\n\n'.join(sumaries)
  template=""" here are the research paper summaries {} and here is the topic list given my the user {}""".format(new_sum,' '.join(topic))
  client = genai.Client(api_key="AIzaSyC6TmVb5Vk5J0r6z0oCmjvNgzbblDKuf3Y")
  response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=cross_systhesis_prompt),
    contents=template
)
  audio = elevenlabs_text_to_speech_and_play(response.text)
  return response.text,audio


def res(query, sort_option):
    global sumaries
    try:
        client = arxiv.Client()

        # Map sort options to arxiv.SortCriterion
        sort_criteria = {
            "Relevance": arxiv.SortCriterion.Relevance,
            "Latest": arxiv.SortCriterion.SubmittedDate,
            "Updated": arxiv.SortCriterion.LastUpdatedDate
        }

        arxiv_search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=sort_criteria.get(sort_option, arxiv.SortCriterion.Relevance)
        )

        formatted_result = ""
        for i, paper in enumerate(client.results(arxiv_search), 1):
            sumaries.append(paper.summary)
            elevenlabs_text_to_speech_and(paper.summary)
            formatted_result += f"""
Paper {i}
{'='*50}
Title: {paper.title}
Authors: {format_authors(paper.authors)}

Summary:
{paper.summary}

Link: {paper.entry_id}
{'='*50}

"""
        if formatted_result:
            return formatted_result.strip()
        else:
            return "No results found!"

    except Exception as e:
        return f"An error occurred: {str(e)}"
# Modified Gradio interface with black text color and improved styling
with gr.Blocks(css="""
    /* Global styles */
    * {

    }

    /* Paper output styling */
    .paper-output {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        padding: 20px;

        border: 1px solid #e0e0e0;
        border-radius: 8px;
            }

    /* Input styling */
    .query-input {
        margin-bottom: 15px;

    }

    /* Button styling */
    .search-button {

        border: 1px solid #d0d0d0;
    }

    /* Tab styling */
    .tab-selected {

        border-bottom: 2px solid black;
    }

    /* Markdown text */
    .md-text {

    }
""") as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("File summarizer"):
            with gr.Row():
                with gr.Column():
                  output=gr.Textbox()
                  audio_output = gr.Audio()   
                with gr.Column():
                    file_input = gr.File(
                        label="Upload a file",
                        elem_classes=["black-text"]
                    )
                    search_button = gr.Button(
                        "Search",
                        variant="primary",
                        elem_classes=["search-button"]
                    )
                    search_button.click(fn=file_upload, inputs=file_input,outputs=[output, audio_output])

                
        with gr.Tab("ArXiv Search"):
            gr.Markdown("""
            ### ArXiv Paper Search
            Enter a search term to find relevant academic papers from ArXiv.
            """, elem_classes=["md-text"])

            with gr.Column():
                # Search input
                text_input = gr.Textbox(
                    placeholder='Enter your search query (e.g., "artificial intelligence", "machine learning")...',
                    label="Search Query",
                    elem_classes=["query-input", "black-text"]
                )

                # Filter section
                with gr.Column(elem_classes=["filter-section"]):
                    gr.Markdown("### Filters", elem_classes=["md-text"])
                    sort_dropdown = gr.Dropdown(
                        choices=["Relevance", "Latest", "Updated"],
                        value="Relevance",
                        label="Sort By",
                        elem_classes=["filter-dropdown"]
                    )

                with gr.Row():
                    search_button = gr.Button(
                        "Search",
                        variant="primary",
                        elem_classes=["search-button"]
                    )

                text_output = gr.Markdown(
                    elem_classes=["paper-output", "black-text"],
                    label="Search Results"
                )

            search_button.click(
                fn=res,
                inputs=[text_input, sort_dropdown],
                outputs=text_output,
                api_name="arxiv_search"
            )
            text_input.submit(
                fn=res,
                inputs=[text_input, sort_dropdown],
                outputs=text_output
            )
            

            gr.Markdown("""
            #### Search Tips:
            - Use specific keywords for better results
            - Enclose exact phrases in quotes
            - Try different combinations of keywords
            - Use filters to sort results by relevance or date
            """, elem_classes=["md-text"])
        with gr.Tab('URL Summarizer'):
          input=gr.Textbox()
          output=gr.Textbox()
          audio_output=gr.Audio()
          search_button = gr.Button(
                        "Summarize",
                        variant="primary",
                        elem_classes=["search-button"]
                    )
          search_button.click(fn=url, inputs=input,outputs=[output,audio_output])
        with gr.Tab('DOI Reference'):
          input=gr.Textbox()
          output=gr.Textbox()
          audio_output=gr.Audio()
          search_button = gr.Button(
                        "Summarize",
                        variant="primary",
                        elem_classes=["search-button"]
                    )
          search_button.click(fn=doi, inputs=input,outputs=[output,audio_output])
        with gr.Tab("Cross-paper synthesis"):
          input=gr.Textbox()
          output=gr.Textbox()
          audio=gr.Audio()
          sumary=gr.Button("Summarize",variant="primary")
          input.submit(fn=topic_adder,inputs=input,outputs=output)
          sumary.click(fn=cross_ana,inputs=None,outputs=[output,audio])



# Launch the demo
demo.launch(debug=True)
