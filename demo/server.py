from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="DeepSeek API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def format_chat_messages(messages: List[ChatMessage]) -> str:
    formatted_messages = []
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted_messages.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            formatted_messages.append(f"Assistant: {msg.content}")
    return "\n".join(formatted_messages) + "\nAssistant:"

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Format the conversation history
        prompt = format_chat_messages(request.messages)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_tokens = inputs.input_ids.shape[1]

        # Generate response
        max_new_tokens = request.max_tokens or 1024
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.n,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode response
        response_text = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
        completion_tokens = len(outputs[0]) - input_tokens

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text.strip()
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=completion_tokens,
                total_tokens=input_tokens + completion_tokens
            )
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek-ai"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
