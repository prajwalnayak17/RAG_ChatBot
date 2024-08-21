from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from llm import llm_chain, chroma_store

app = FastAPI()

@app.post("/faq/")
async def faq_query(query: str):
    try:
        result = chroma_store.similarity_search(query)
        if not result:
            return JSONResponse(status_code=404, content={"error": "No relevant information found."})
        return {"answer": result[0].page_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/")
async def chat_with_bot(user_input: str):
    response = await llm_chain.arun(user_input=user_input)
    return {"response": response}


@app.post("/service/")
async def service_inquiry(service_type: str, name: str, email: str, phone: str):
    # Process the service inquiry and store the user's details
    # You can integrate this with your database logic here
    return {"message": f"Service inquiry received for {service_type}. We'll contact you soon, {name}."}