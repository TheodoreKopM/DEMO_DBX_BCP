from databricks.agent import Agent, AgentConfig
from databricks.agent.tools import VectorSearchRetrieverTool
from databricks.vector_search.client import VectorSearchClient
import mlflow
from typing import List, Dict, Any
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda

class RAGAgent:
    def __init__(
        self,
        vector_search_endpoint_name: str,
        vector_search_index_name: str,
        llm_endpoint_name: str,
        schema: Dict[str, str],
        retriever_config: Dict[str, Any],
        llm_config: Dict[str, Any]
    ):
        self.vs_client = VectorSearchClient(disable_notice=True)
        self.vs_index = self.vs_client.get_index(
            endpoint_name=vector_search_endpoint_name,
            index_name=vector_search_index_name
        )
        
        # Configure the Vector Search Retriever Tool
        self.retriever_tool = VectorSearchRetrieverTool(
            index=self.vs_index,
            text_column=schema["chunk_text"],
            columns=[
                schema["primary_key"],
                schema["chunk_text"],
                schema["document_uri"]
            ],
            search_kwargs=retriever_config["parameters"]
        )
        
        # Set up MLflow tracking
        mlflow.langchain.autolog()
        mlflow.models.set_retriever_schema(
            primary_key=schema["primary_key"],
            text_column=schema["chunk_text"],
            doc_uri=schema["document_uri"]
        )
        
        # Configure query rewriting for multi-turn conversations
        self.query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

        self.query_rewrite_prompt = PromptTemplate(
            template=self.query_rewrite_template,
            input_variables=["chat_history", "question"]
        )
        
        # Configure the Agent
        self.agent_config = AgentConfig(
            llm_endpoint=llm_endpoint_name,
            llm_parameters=llm_config["llm_parameters"],
            system_prompt=llm_config["llm_system_prompt_template"],
            tools=[self.retriever_tool]
        )
        
        # Initialize the Agent
        self.agent = Agent(config=self.agent_config)
        
        # Set up the RAG chain
        self.chain = self._build_rag_chain()
    
    def _format_context(self, docs):
        """Format the retrieved documents into a context string."""
        chunk_template = self.retriever_tool.chunk_template
        chunk_contents = [
            chunk_template.format(
                chunk_text=d.page_content,
                document_uri=d.metadata[self.retriever_tool.schema["document_uri"]]
            )
            for d in docs
        ]
        return "".join(chunk_contents)
    
    def _extract_user_query(self, messages: List[Dict[str, str]]) -> str:
        """Extract the latest user message from the conversation."""
        return next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            None
        )
    
    def _extract_chat_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract the chat history excluding the latest message."""
        return messages[:-1]
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Format chat history for the query rewrite prompt."""
        history = self._extract_chat_history(messages)
        if not history:
            return ""
        
        formatted_history = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    def _build_rag_chain(self):
        """Build the RAG chain with query rewriting for multi-turn conversations."""
        return (
            {
                "question": itemgetter("messages") | RunnableLambda(self._extract_user_query),
                "chat_history": itemgetter("messages") | RunnableLambda(self._extract_chat_history),
            }
            | RunnablePassthrough()
            | {
                "context": RunnableBranch(
                    (
                        lambda x: len(x["chat_history"]) > 0,
                        self.query_rewrite_prompt | self.agent.llm | StrOutputParser()
                    ),
                    itemgetter("question")
                )
                | self.retriever_tool
                | RunnableLambda(self._format_context),
                "question": itemgetter("question")
            }
            | self.agent
        )
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Process a chat message and return the agent's response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: The agent's response
        """
        if not messages:
            return "No messages provided."
        
        # Get the agent's response using the RAG chain
        response = self.chain.invoke({"messages": messages})
        return response.content
    
    def log_model(self, model_name: str):
        """
        Log the agent model to MLflow.
        
        Args:
            model_name: Name to use for the logged model
        """
        mlflow.langchain.log_model(
            lc_model=self.chain,
            artifact_path=model_name,
            registered_model_name=model_name
        ) 