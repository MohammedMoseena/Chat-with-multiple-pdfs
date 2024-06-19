Steps that are followed

1. Create virtual environment and activate virtual environment
2. Install all the dependencies mentioned in requirments.txt
3. Get GOOGLE_API_KEY from "https://aistudio.google.com/app/apikey"
4. Make sure the project file has git setup(for eg: .gitattributes, .gitignore)
5. commit and push all the changes to remote repository

END TO END Process:

Upload Multiple PDFs: Users upload multiple PDF files using Streamlit's file uploader.

Extract Text from PDFs: The text from all uploaded PDFs is extracted and concatenated into a single string (text).

Split Text into Chunks: The concatenated text is split into smaller chunks based on chunk_size and chunk_overlap using RecursiveCharacterTextSplitter. This step ensures that large text data is divided into manageable segments for further processing.

Generate Embeddings: Each chunk of text is then converted into embeddings using GoogleGenerativeAIEmbeddings, which presumably leverages Google's Generative AI models (models/embedding-001). These embeddings capture semantic information about the text chunks.

Store Vectors in FAISS Database: The embeddings are stored in a FAISS index (faiss_index) using FAISS.from_texts(text_chunks, embedding=embeddings). This index allows for efficient similarity searches based on the embeddings.

User Input: Question Answering:

When a user inputs a question (user_question), it undergoes a similarity search against the embeddings stored in the FAISS index (faiss_index) to find the most relevant chunks or documents (docs) related to the question.
Generate Response:

The relevant documents (docs) and the user question (user_question) are then passed through a prompt_template that structures the input for a question-answering model (ChatGoogleGenerativeAI).
The prompt_template ensures that the model understands the context (context) and the question (question) in a structured format.
Finally, the structured prompt (prompt_template) is passed to the ChatGoogleGenerativeAI model (gemini-pro) to generate a relevant response (output_response).



APPENDIX:

ABOUT CHUNKS

Chunk Size:

Impact: The chunk_size parameter determines the maximum number of characters that each chunk of text will contain.
Effect on Output:
Larger Chunk Size: Larger chunks may result in fewer segments overall, potentially leading to more general or broader representations of the text content. This can reduce the computational overhead but might sacrifice detail in the analysis.
Smaller Chunk Size: Smaller chunks provide more fine-grained segmentation, allowing for more detailed analysis of the text content. However, this can increase computational complexity and memory usage.

Chunk Overlap:

Impact: The chunk_overlap parameter specifies how many characters each chunk overlaps with its neighboring chunks.
Effect on Output:
Higher Overlap: A higher overlap between chunks ensures that important contextual information is retained across adjacent segments. This can enhance the coherence and contextuality of the extracted segments.
Lower Overlap: Lower overlap results in more distinct and independent segments. While this may reduce redundancy, it could potentially lead to fragmented context representation, especially in cases where text continuity is important.
Optimization:

Finding the Balance: Choosing appropriate values for chunk_size and chunk_overlap involves finding a balance between computational efficiency, granularity of analysis, and the need to maintain contextual coherence. It often requires experimentation and consideration of specific requirements and constraints of the text processing task at hand.

ABOUT TEMPERATURE:
The temperature parameter adjusts the level of randomness in the generated outputs. A lower temperature tends to produce more deterministic and conservative responses, often selecting more common or likely outputs. On the other hand, a higher temperature introduces more variability and creativity, resulting in more diverse but potentially less coherent or relevant outputs.
