# query_doc
A RAG Based application to query a PDF.

Dependencies<br/><br/>
The project uses the following dependencies:<br/>

* ***inngest***: Used to orchestrate and run event-driven workflows and background tasks reliably.<br>
* ***qdrant***: Used as a vector database to store, search, and manage high-dimensional embeddings, enabling efficient similarity search.<br>
* ***all-MiniLM-L6-v2***: A pre-trained SentenceTransformer model used to converts text into dense vector embeddings.<br>
* ***llama3:8b***: Used to generate text based on a prompt.<br>
* ***streamlit***: Used to create a user-friendly web application for querying and adding documents.<br>

## Usage

1. Setup qdrant vector database on docker:

   ```bash
   docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
   ```
   
2. Setup Ollama on docker:

   ```bash
   docker run -d --name ollama --gpus all -p 11434:11434 -v "$(pwd)/ollama:/ollama" ollama/ollama
   ```
   
3. Install necessary dependencies:

   ```bash
   uv add <dep1>, <dep2>
   ```
   
4. Run the app:

   ```bash
   uv run uvicorn main:app
   ```
   
4. Start Inngest server using npx:

   ```bash
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
   ```
   
5. Run streamlit application:

   ```bash
   uv run streamlit run .\streamlit_app.py
   ```
   
## Demo

### Streamlit Application

<img src="/results/result1.png" />

###

<img src="/results/result2.png" />

###

### Inngest 

<img src="/results/inngest1.png" />

###

<img src="/results/inngest2.png" />

###


