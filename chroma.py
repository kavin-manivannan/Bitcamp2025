import chromadb
import csv

data_set = r"C:\Users\abith\Downloads\Preprocessed_Data.txt"

client = chromadb.Client()

collection = client.create_collection(name='resume_embeddings')

documents = []
metadatas = []
ids = []

with open(data_set, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    for idx, row in enumerate(reader):
        if len(row) == 2:
            category, text = row
            category = category.strip()
            text = text.strip()

            documents.append(text)
            metadatas.append({"Category": category, "Text": text})
            ids.append(f"doc_{idx}")

max_documents = 1000
batch_size = 500

documents_added = documents[:max_documents]
metadatas_added = metadatas[:max_documents]
ids_added = ids[:max_documents]

for i in range(0, len(documents_added), batch_size):
    end = i + batch_size
    collection.add(documents=documents_added[i:end], metadatas=metadatas_added[i:end], ids=ids_added[i:end])
