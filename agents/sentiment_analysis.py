import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Cargar documentos desde un archivo de texto
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/text.txt")
text_documents = loader.load()

# Dividir los documentos en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
doc_txt = text_splitter.split_documents(text_documents)

# Crear embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

# Crear el vectorstore con Chroma
db = Chroma.from_documents(doc_txt, embeddings)

# Configurar el modelo LLM
llm = OpenAI(openai_api_key=api_key)

# Crear el prompt para análisis de sentimientos
prompt_template = """
Eres un asistente experto en análisis de sentimientos. Tu tarea es analizar un texto y determinar su sentimiento como 
Positivo, Negativo o Neutral. Además, debes proporcionar una explicación clara y detallada sobre por qué clasificaste 
el texto de esa manera. También, si detectas emociones específicas (como felicidad, enojo, frustración, alegría, 
tristeza, etc.), identifícalas y explícalas.

Sigue las siguientes reglas:
1. Lee atentamente el texto y analiza el tono general.
2. Considera palabras clave, emociones implícitas y contexto.
3. Clasifica el texto como:
   - **Positivo**: Cuando el texto tiene un tono optimista, alegre o elogioso.
   - **Negativo**: Cuando el texto refleja críticas, frustraciones o insatisfacciones.
   - **Neutral**: Cuando el texto no tiene emociones claras o es equilibrado.

4. Proporciona una explicación de al menos 2-3 oraciones sobre cómo llegaste a la conclusión.

Ejemplo 1:
Texto: "El servicio fue increíble, me sentí muy bien atendido, y definitivamente recomendaré este lugar."
Análisis:
Sentimiento: Positivo
Explicación: El autor utiliza palabras como "increíble" y "muy bien atendido", lo que indica satisfacción y alegría. 
Además, la intención de recomendar el lugar refuerza el tono positivo.

Ejemplo 2:
Texto: "El producto llegó tarde y encima estaba dañado. Nadie respondió a mis quejas. Estoy muy decepcionado."
Análisis:
Sentimiento: Negativo
Explicación: El texto refleja frustración y decepción a través de expresiones como "llegó tarde", "estaba dañado" y 
"nadie respondió". Esto indica una experiencia negativa con el producto y el servicio al cliente.

Ejemplo 3:
Texto: "El servicio estuvo bien, pero nada excepcional. No tengo quejas importantes, aunque tampoco es algo que recomendaría con entusiasmo."
Análisis:
Sentimiento: Neutral
Explicación: El autor no expresa emociones fuertes ni críticas importantes. Aunque no está particularmente impresionado, 
tampoco refleja insatisfacción.

Ahora analiza el siguiente texto y sigue el mismo formato para proporcionar tu análisis.
Texto: {query}
"""
prompt = PromptTemplate(input_variables=["query"], template=prompt_template)

def analyze_sentiment(query):
    """Realiza una búsqueda por similitud y analiza el sentimiento."""
    try:
        # Buscar contexto relevante en el vectorstore
        print(f"Realizando búsqueda por similitud para: {query}")
        results = db.similarity_search(query)
        context = results[0].page_content if results else "No se encontró contexto relevante."

        # Formar el prompt final
        final_prompt = prompt.format(query=query, context=context)

        # Enviar al modelo para análisis
        print(f"Contexto encontrado: {context}")
        response = llm(final_prompt)
        return response
    except Exception as e:
        print(f"Error en el análisis: {str(e)}")
        return f"Error en el análisis: {str(e)}"