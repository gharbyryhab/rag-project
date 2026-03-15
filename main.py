from dotenv import load_dotenv
import os
import time
import shutil
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# ── Étape 1 : Extraction avec métadonnées par page ────────
PDF_PATH = "data/document.pdf"
PDF_NAME = os.path.basename(PDF_PATH)

elements = partition_pdf(PDF_PATH)

# Regrouper le texte par page
pages = {}
for el in elements:
    page = el.metadata.page_number if el.metadata.page_number else 1
    if page not in pages:
        pages[page] = []
    pages[page].append(str(el))

# Créer un Document par page
docs_par_page = []
for page_num, textes in pages.items():
    docs_par_page.append(Document(
        page_content="\n".join(textes),
        metadata={"source": PDF_NAME, "page": page_num}
    ))

print(f"✅ {len(docs_par_page)} pages extraites")

# ── Étape 2 : Chunking ────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs_par_page)
print(f"✅ {len(chunks)} chunks avec métadonnées")
print(f"   Exemple → page: {chunks[0].metadata['page']}, source: {chunks[0].metadata['source']}\n")

# ── Étape 3 : Réinitialiser ChromaDB ─────────────────────
if os.path.exists("vectorstore/"):
    shutil.rmtree("vectorstore/")
    print("🗑️  Ancien vectorstore supprimé")

# ── Étape 4 : Embeddings par batch avec délai ─────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
    transport="rest"
)

# Ingestion par petits batches pour éviter le quota RPM
BATCH_SIZE = 10
print(f"⏳ Ingestion par batches de {BATCH_SIZE}...")

vectorstore = None
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="vectorstore/"
        )
    else:
        vectorstore.add_documents(batch)
    print(f"   Batch {i//BATCH_SIZE + 1} ingéré ({min(i+BATCH_SIZE, len(chunks))}/{len(chunks)})")
    time.sleep(3)  # pause pour respecter le quota RPM

print(f"\n✅ ChromaDB rechargé avec métadonnées !\n")

# ── Étape 5 : LLM + Chain RAG ─────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

prompt = PromptTemplate.from_template("""
Tu es un assistant pédagogique. Réponds à la question en te basant
uniquement sur le contexte fourni. Si la réponse n'est pas dans
le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Étape 6 : Boucle interactive avec sources ─────────────
print("=" * 50)
print("   🤖 Assistant RAG — Bases de Données")
print("   Tape 'exit' pour quitter")
print("=" * 50)

while True:
    print()
    question = input("❓ Ta question : ").strip()

    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        print("👋 Au revoir !")
        break

    docs = retriever.invoke(question)
    reponse = rag_chain.invoke(question)

    print("\n🤖 Réponse :\n")
    print(reponse)

    print("\n📚 Sources utilisées :")
    sources_vues = set()
    for doc in docs:
        cle = (doc.metadata['source'], doc.metadata['page'])
        if cle not in sources_vues:
            print(f"   • {doc.metadata['source']} — page {doc.metadata['page']}")
            sources_vues.add(cle)

    print("\n" + "-" * 50)