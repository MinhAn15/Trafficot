import logging
import sys
import os
import openai
import numexpr as ne
import nest_asyncio
from llama_index.core import ListIndex, ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, PromptHelper
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
#import chromadb
#from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import RecursiveRetriever
#from llama_index.core import get_response_synthesizer
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector, PydanticMultiSelector, PydanticSingleSelector
from llama_index.core.response.pprint_utils import pprint_response

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '4'



nest_asyncio.apply()

def index_export(user_input):
    # Document for Vector index
    required_exts = [".txt"]
    filename_fn = lambda filename: {"doc_id": filename.split('\\')[-1]}
    documents = SimpleDirectoryReader(input_dir="./data/", recursive=True, required_exts=required_exts, filename_as_id=True,file_metadata=filename_fn).load_data()



    # Call Vector Index
    # vector_index = VectorStoreIndex.from_documents(docs_lst)
    service_context = ServiceContext.from_defaults(chunk_size=256)
    nodes = service_context.node_parser.get_nodes_from_documents(documents)
    list_index = ListIndex(nodes)

    list_query_engine = list_index.as_query_engine(
        response_mode = "tree_summarize", use_async = True
    )






    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
        verbose=True
    )
    callback_manager = CallbackManager([token_counter])


    llm = OpenAI(model="gpt-3.5-turbo")
    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=80)
    Settings.num_output = 2048
    Settings.context_window = 4096
    Settings.transformations = [SentenceSplitter(chunk_size=256)]
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    Settings.callback_manager = CallbackManager([token_counter])



    # chroma_client = chromadb.EphemeralClient()
    # db = chromadb.PersistentClient(path="./storage/chroma")
    # chroma_collection = db.get_or_create_collection("traffic_db")
    # chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)





    llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    #define prompt helper
    #set maximum input size
    max_input_size = 13000
    #set number of output tokens
    num_output = 2048
    #set maimum chunk overlap
    max_chunk_overlap = 0.8
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm=llm, prompt_helper=prompt_helper, callback_manager=callback_manager)

    try:
        #rebuild storage index
        storage_context = StorageContext.from_defaults(persist_dir='./storage/')#, vector_store=chroma_store)
        #load index
        index = VectorStoreIndex.from_documents(documents, service_context=service_context, Settings=Settings,storage_context=storage_context)
    
        print('loading from disk')
    except:
        index = VectorStoreIndex.from_documents(documents, service_context=service_context, Settings=Settings)#,storage_context=storage_context)
        index.storage_context.persist(persist_dir='./storage/')
        print('persisting to disk')





    text_qa_template_str = (
         "You are AI assistant of smart traffic system and be living in Ho Chi Minh city.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Prior the context information to "
    "answer the question: {query_str}\n"
    "3 Rules to follow:\n"
    "1.If the context information isn't helpful, use vietnamese to said that information is not available. After that response the suggestion by your own knowledge.\n"
    "2.Always Use vietnamese to answer.\n"
    "3.Use bullet point and human's natural tone.")
        

    text_qa_template = PromptTemplate(text_qa_template_str)




    retriever = VectorIndexRetriever(
        index = index,
        similarity_top_k=3,
    )
    s_processor = SimilarityPostprocessor(similarity_cutoff=0.83)
    k_processor = KeywordNodePostprocessor(
        exclude_keywords = ["chết", "tử vong"],
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )

    query_engine_retriever = RetrieverQueryEngine.from_args(
        retriever=retriever,
    node_postprocessors=[k_processor,s_processor],
    response_synthesizer=response_synthesizer,
    text_qa_template=text_qa_template,
    #vector_store=chroma_store,
    )





    list_tool = QueryEngineTool.from_defaults(
        query_engine = list_query_engine,
        description="Tổng hợp thông tin của thành phố Hồ Chí Minh"
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine_retriever,
        description="Thông tin chi tiết các quận của thành phố Hồ Chí Minh"
    )



    query_engine_pydantic = RouterQueryEngine(
        selector=PydanticMultiSelector.from_defaults(),
        query_engine_tools=[
            list_tool,
            vector_tool
        ],   
    )

    response = index.as_query_engine(query_engine=query_engine_pydantic).query(user_input)
    return response





# #pprint_response(response,show_source=True)
# print('Embedding tokens: ', token_counter.total_embedding_token_count, '\n',
#     'LLM prompts: ', token_counter.prompt_llm_token_count, '\n',
#     'LLM completitions: ', token_counter.completion_llm_token_count, '\n',
#     'Total LLM token count: ', token_counter.total_llm_token_count, '\n')
# print(response)
