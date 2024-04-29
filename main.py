from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import rag_chain

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.post("/answer/")
async def get_answer(request: QuestionRequest):
    try:
        result = rag_chain.invoke(request.question)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, deatils=str(e))