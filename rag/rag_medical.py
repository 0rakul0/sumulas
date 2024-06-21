from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
import nest_asyncio
import chromadb

# Aplicar nest_asyncio
nest_asyncio.apply()

# Chamar o modelo pré-treinado
llm = Ollama(model="llama3:latest", request_timeout=420)
Settings.llm = llm

# Função para embutir o texto
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text: str) -> List[float]:
    embedding = model.encode([text], convert_to_tensor=True)
    return embedding.cpu().numpy()[0].tolist()

# Chamar o chromadb
chroma_client = chromadb.Client()

# Verificar a criação do banco
collection_name = "casos_medicos"
collections = chroma_client.list_collections()

# Chamar o chroma se não tiver ele gera um
if collection_name in [col.name for col in collections]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(collection_name)

# Adicionar dados ao banco casos novos
medical_cases = [
    "caso 1...paciente com febre alta, dor de cabeça e dor muscular. Diagnóstico: Dengue. Tratamento: Hidratação, Repouso",
    "caso 2...paciente com tosse seca, falta de ar e febre. Diagnóstico: COVID-19. Tratamento: Isolamento, Antitérmicos, Hidratação",
    "caso 3...paciente com dor abdominal intensa, febre e náusea. Diagnóstico: Apendicite. Tratamento: Cirurgia (Apendicectomia), Antibióticos",
    "caso 4...paciente com dor torácica, sudorese e falta de ar. Diagnóstico: Infarto do Miocárdio. Tratamento: Angioplastia, Medicamentos Anticoagulantes",
    "caso 5...paciente com perda de peso, aumento da sede e urinação frequente. Diagnóstico: Diabetes Mellitus. Tratamento: Insulina, Dieta Controlada",
    "caso 6...paciente com coceira intensa, lesões na pele e febre baixa. Diagnóstico: Varicela (Catapora). Tratamento: Antitérmicos, Antipruriginosos",
    "caso 7...paciente com dor de garganta, febre e placas esbranquiçadas nas amígdalas. Diagnóstico: Amigdalite Bacteriana. Tratamento: Antibióticos, Analgésicos",
    "caso 8...paciente com tosse persistente, febre e suores noturnos. Diagnóstico: Tuberculose. Tratamento: Antibióticos (RIPE: Rifampicina, Isoniazida, Pirazinamida, Etambutol)",
    "caso 9...paciente com dor ao urinar, febre e dor lombar. Diagnóstico: Infecção do Trato Urinário. Tratamento: Antibióticos, Hidratação",
    "caso 10...paciente com rigidez no pescoço, febre alta e dor de cabeça. Diagnóstico: Meningite. Tratamento: Antibióticos, Hospitalização"
]

for i, case in enumerate(medical_cases):
    embedding = embed_text(case)
    collection.add(embeddings=[embedding], ids=[f"case_{i}"], metadatas=[{"content": case}])

# Função para buscar casos médicos similares
def procura_caso_similar(query: str, top_k: int = 3) -> List[str]:
    try:
        query_embedding = embed_text(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        similar_cases = [metadata["content"] for metadata in results["metadatas"][0]]
        return similar_cases
    except Exception as e:
        st.error(f"Erro ao buscar casos similares: {e}")
        return []

# Define a ferramenta para o Agente
procura_caso_similar_tool = FunctionTool.from_defaults(fn=procura_caso_similar)

# Função para gerar diagnósticos
def gerar_diagnostico_e_tratamento(novo_caso: str, casos_similares: List[str]) -> str:
    input_text = f"Sintomas do novo caso médico:{novo_caso}\n\nCasos similares: {' '.join(casos_similares)}"
    messages = [
        ChatMessage(
            role="system", content="Você é um especialista em medicina e fornece diagnósticos e tratamentos de acordo com os sintomas do paciente."
        ),
        ChatMessage(role="user", content=input_text),
        ChatMessage(role="user", content="Você só deve fornecer diagnóstico e tratamento se baseando nos sintomas, diagnósticos e tratamentos do caso médico mais similar aos sintomas no novo caso médico.\n"
                                          "Não adicione nenhum texto a mais no texto do novo caso.\n"
                                          "Na sua resposta deve somente constar a lista dos casos similares e a explicação para o diagnóstico e tratamento.")
    ]
    try:
        resp = llm.chat(messages=messages)
        return resp
    except Exception as e:
        st.error(f"Erro ao gerar diagnóstico e tratamento: {e}")
        return ""

gerar_diagnostico_e_tratamento_tool = FunctionTool.from_defaults(fn=gerar_diagnostico_e_tratamento)

# Chamando o agente para se comportar como pedido usando as ferramentas criadas
agent = ReActAgent.from_tools(
    [procura_caso_similar_tool, gerar_diagnostico_e_tratamento_tool],
    llm=llm,
    verbose=True,
)

# Interface com o Streamlit
st.title("Assistente de Diagnóstico e Tratamento Médico")

novo_caso = st.text_area("Insira os sintomas e informações do novo caso")
if st.button("Buscar casos similares e gerar diagnóstico"):
    if novo_caso:
        with st.spinner("Buscando tratamento..."):
            casos_similares = procura_caso_similar(novo_caso)
            if casos_similares:
                st.subheader("Casos Similares Encontrados")
                for caso in casos_similares:
                    st.write(caso)
                diagnostico_e_tratamento = agent.chat(f"procure pelos casos médicos mais similares a esse novo caso {novo_caso} visando usá-los para obter o diagnóstico e tratamento do novo caso")
                st.subheader("Diagnóstico e Tratamento Proposto")
                st.write(diagnostico_e_tratamento.response)  # Ajustar para acessar a resposta corretamente
            else:
                st.error("Nenhum caso similar encontrado.")
    else:
        st.error("Por favor, informe os sintomas e informações do novo caso.")
