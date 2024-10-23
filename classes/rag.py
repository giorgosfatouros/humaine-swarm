import os
from openai import AsyncOpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from utils.helper_functions import setup_logging
import logging
from openai import OpenAI
from typing import List
from datetime import datetime
from literalai import AsyncLiteralClient
import tiktoken
import asyncio

lai = AsyncLiteralClient()
lai.instrument_openai()

logger = setup_logging('RAG', level=logging.ERROR)

class RAG:
    def __init__(self, api_key=os.getenv('API_KEY', None), model="gpt-4o-mini", embed_model="text-embedding-3-small", temperature=0, max_tokens=3000, top_p=1, frequency_penalty=1, presence_penalty=0.8):
        self.api_key = api_key
        self.tokenizer = tiktoken.encoding_for_model(model)  # Adjust the model as needed

        self.client = AsyncOpenAI(api_key=api_key)
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.embed_model = embed_model
        self.index = self.pc.Index("macro")

        self.settings = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

    # async def get_embeddings(self, text, **kwargs) -> List[float]:
    #     # replace newlines, which can negatively affect performance.
    #     text = text.replace("\n", " ")
    #     client = OpenAI(api_key=self.api_key)
    #     response = await client.embeddings.create(input=[text], model=self.embed_model, **kwargs)
    #     return response.data[0].embedding
    
    async def get_embeddings(self, text, **kwargs) -> List[float]:
    # Replace newlines
        text = text.replace("\n", " ")
        response = await self.client.embeddings.create(
            input=[text],
            model=self.embed_model,
            **kwargs
        )
        return response.data[0].embedding

    def prepare_filter(self, filters: dict = {}) -> dict:
        pinecone_filter = {}
        if filters:
            if 'date_range' in filters:
                try:
                    # Split the date range if it has a " to " separator
                    date_parts = filters['date_range'].split(' to ')

                    if len(date_parts) == 1:
                        # If only one date or year is provided, assume it as the start date and generate an end date
                        start_date_str = date_parts[0]
                        if len(start_date_str) == 4:  # Year only provided
                            start_date_str += "-01-01"
                            end_date_str = start_date_str[:4] + "-12-31"
                        else:
                            end_date_str = start_date_str
                    elif len(date_parts) == 2:
                        # If both start and end dates are provided, process accordingly
                        start_date_str, end_date_str = date_parts
                        # Handle year-only inputs by adding default month and day
                        if len(start_date_str) == 4:
                            start_date_str += "-01-01"
                        if len(end_date_str) == 4:
                            end_date_str += "-12-31"
                    else:
                        raise ValueError("Invalid date range format.")

                    # Convert to datetime and then to timestamp
                    pinecone_filter['Timestamp'] = {
                        "$gte": int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp()),
                        "$lte": int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp())
                    }
                except ValueError as e:
                    # Log the error and do not add the date filter
                    logger.error(f"Error parsing date range: {str(e)}")
                    logger.warning("Leaving date range filter as None due to invalid input format.")


        return pinecone_filter
    
    async def retrieve_documents(self, query, filters):
        xq = await self.get_embeddings(query)
        pinecone_filter = self.prepare_filter(filters)
        logger.info(f"pinecone_filter: {pinecone_filter}")

        # Use asyncio.to_thread to run the blocking call in a separate thread
        query_result = await asyncio.to_thread(
            self.index.query,
            vector=xq,
            top_k=5,
            include_metadata=True,
            filter=pinecone_filter
        )
        results = query_result['matches']

        # Sort results by Timestamp in descending order (most recent first)
        results.sort(key=lambda x: x['metadata']['Timestamp'], reverse=True)

        # Process results to limit tokens and prioritize most recent info
        max_context_tokens = 1500
        context_texts = []
        total_tokens = 0
        for res in results:
            text = res['metadata']['text']
            date = res['metadata']['Date']
            text += f" [Source: [{res['metadata']['Organization']}]({res['metadata']['Link']})]"
            # if res['metadata']['Organization'] in ['FED', 'ECB']:
            #     text += f" [Source: [{res['metadata']['Organization']}]({res['metadata']['Link']})]"
            # else:
            #     text += f" [Publisher: {res['metadata']['Organization']}]"
            text_tokens = len(self.tokenizer.encode(text))
            if total_tokens + text_tokens > max_context_tokens:
                continue
            context_texts.append(f"Published on {date}: {text}")
            total_tokens += text_tokens
        chunks = "\n--\n".join(context_texts)
        logger.info(f"RAG: {chunks}")
        return chunks



    # async def generate_response(self, message_history, augmented_prompt):
    #     message_history.append({"role": "user", "content": augmented_prompt})
    #     stream = await self.client.chat.completions.create(
    #         messages=message_history,
    #         stream=True,
    #         **self.settings
    #     )
    #     return stream