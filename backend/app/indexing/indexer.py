import os
import csv
from typing import Dict, Union
from pydantic import BaseModel
import chromadb
from app.clients import async_openai_client
import asyncio
from dataclasses import dataclass


DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

class Data(BaseModel):
    """Base model for medical code data.
    
    Attributes:
        code: The medical code identifier.
        text: The human-readable description of the code.
    """
    code: str
    text: str


@dataclass(frozen=True)
class Datasets:
    """Constants for different medical vocabulary datasets."""
    ICD = "icd"
    CPT = "cpt"
    LOINC = "loinc"
    ATC = "atc"


class Indexer:

    def __init__(self) -> None:
        self.client_db = chromadb.PersistentClient()

    def get_icd_data(self) -> list[Data]:
        file_path = os.path.join(DATA_ROOT, 'icd10cm_codes_2026.txt')
        
        data = []
        codes = set()
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        code, text = parts
                        if code not in codes:
                            data.append(Data(text=text, code=code))
                            codes.add(code)
        return data
    
    def get_atc_data(self) -> list[Data]:
        file_path = os.path.join(DATA_ROOT, 'WHO ATC-DDD 2024-07-31.csv')
        
        data = []
        codes = set()
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                code = row['atc_code']
                text = row['atc_name']
                if code and text and code not in codes:
                    data.append(Data(text=text, code=code))
                    codes.add(code)
        return data
    
    def get_cpt_data(self) -> list[Data]:
        file_path = os.path.join(DATA_ROOT, '2025_DHS_Code_List_Addendum_11_26_2024.txt')
        
        data = []
        codes = set()
        with open(file_path, 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('"') and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        code, text = parts
                        code = code.strip()
                        text = text.strip()
                        if code and text and code not in codes:
                            data.append(Data(text=text, code=code))
                            codes.add(code)
        
        return data
        
    def get_loinc_data(self) -> list[Data]:
        file_path = os.path.join(DATA_ROOT, 'LoincTableCore.csv')
        
        data = []
        codes = set()
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                code = row['LOINC_NUM']
                text = row['LONG_COMMON_NAME']
                if code and text and code not in codes:
                    data.append(Data(text=text, code=code))
                    codes.add(code)
        
        return data
    
    async def embed_batch(self, batch: list[Data]) -> list[list[float]]:
        """Generate embeddings for a batch of medical codes.
        
        Args:
            batch: List of Data objects to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            Exception: If embedding generation fails.
        """
        try:
            response = await async_openai_client.embeddings.create(
                input=[item.text for item in batch],
                model="text-embedding-3-small"
            )
            return [res.embedding for res in response.data]
        except Exception as e:
            print(f"Embedding failed for batch (size {len(batch)}): {e}")
            raise

    async def embed_data(self, data: list[Data]) -> list[list[float]]:
        """Generate embeddings for all data in batches.
        
        Args:
            data: List of Data objects to embed.
            
        Returns:
            List of embedding vectors for all input data.
        """
        batch_size: int = 1000
        tasks: List[asyncio.Task] = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            tasks.append(asyncio.create_task(self.embed_batch(batch)))
        
        batch_results = await asyncio.gather(*tasks)
        embeddings = []
        for batch_embeddings in batch_results:
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def index_data(self, data: list[Data], collection: str) -> None:
        """Index medical code data into ChromaDB collection.
        
        Args:
            data: List of Data objects to index.
            collection: Name of the ChromaDB collection.
        """

        index = self.client_db.get_or_create_collection(name=collection)
        if index.count() != 0:
            return
        
        # Remove duplicates by code
        unique_data: Dict[str, Data] = {}
        for item in data:
            if item.code not in unique_data:
                unique_data[item.code] = item
        data = list(unique_data.values())
        
        embeddings = await self.embed_data(data)

        # Add in batches to respect ChromaDB limits
        batch_size: int = 5000
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            index.add(
                documents=[d.text for d in batch_data],
                embeddings=batch_embeddings,
                ids=[d.code for d in batch_data]
            )

    async def index_all(self) -> None:
        """Index all medical vocabulary datasets concurrently."""
        icd_data = self.get_icd_data()
        cpt_data = self.get_cpt_data()
        loinc_data = self.get_loinc_data()
        atc_data = self.get_atc_data()

        tasks = [
            asyncio.create_task(self.index_data(icd_data, Datasets.ICD)),
            asyncio.create_task(self.index_data(cpt_data, Datasets.CPT)), 
            asyncio.create_task(self.index_data(loinc_data, Datasets.LOINC)), 
            asyncio.create_task(self.index_data(atc_data, Datasets.ATC)) 
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed: {result}")
            else:
                print(f"Task {i} completed successfully")

    async def search(self, query: str, collection: str, max_results: int = 15) -> list[Data]:
        """Search for medical codes using vector similarity.
        
        Args:
            query: Search query text.
            collection: ChromaDB collection name to search in.
            max_results: Maximum number of results to return.
            
        Returns:
            List of Data objects matching the query.
        """

        index = self.client_db.get_or_create_collection(name=collection)

        response = await async_openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        results = index.query(
            query_embeddings=[response.data[0].embedding],
            n_results=max_results,
            include=["documents"]
        )

        docs = results["documents"][0]  # ChromaDB returns nested list
        ids = results["ids"][0]
        data = [Data(code=code, text=doc) for code, doc in zip(ids, docs)]
        return data
    


    


        