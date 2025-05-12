import mlflow
import yaml
from rag_agent import RAGAgent

# Load configuration
with open("rag_agent_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize the RAG agent
agent = RAGAgent(
    vector_search_endpoint_name=config["chain_config"]["databricks_resources"]["vector_search_endpoint_name"],
    vector_search_index_name=config["chain_config"]["retriever_config"]["vector_search_index"],
    llm_endpoint_name=config["chain_config"]["databricks_resources"]["llm_endpoint_name"],
    schema=config["chain_config"]["retriever_config"]["schema"],
    retriever_config=config["chain_config"]["retriever_config"],
    llm_config=config["chain_config"]["llm_config"]
)

# Example conversation
messages = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG is a technique..."},
    {"role": "user", "content": "How do I implement it?"}
]

# Get response from the agent
response = agent.chat(messages)
print(response)

# Log the model to MLflow
agent.log_model("rag_agent_model") 