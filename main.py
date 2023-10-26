import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from llama_index import (
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.prompts import PromptTemplate


qa_prompt_str = (
    "あなたはとあるプロダクトのセキュリティ担当者です。\n"
    "プロダクトのセキュリティに関するお客様からのご質問に回答する責任を持っています。\n"
    "事前知識ではなく、常に提供されたコンテキストを使用してクエリに回答してください。\n"
    "従うべきいくつかのルール:\n"
    "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
    "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。\n"
    "3. 必ず日本語で回答してください。\n"
    "4. 貴社はプロダクトを提供するあなたが所属する会社、弊社はプロダクトを利用してくれる会社、他社はプロダクトを利用している他の会社として認識してください。\n"
    "コンテキストは以下のとおりです。\n"
    "なお、コンテキストのQ. に続く文字は過去の質問で、A. に続く文字は過去の回答です。過去の回答がはい、か、いいえのみの場合は、今回の回答もはい、か、いいえだけで答えなさい。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "事前知識ではなく提供されたコンテキストのみを使用してクエリに答えなさい。\n"
    "クエリ: {query_str}\n"
)
refine_template_str = (
    "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。\n"
    "1. 新しいコンテキストを使用して元の回答を**書き直す**。\n"
    "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n"
    "3. 必ず日本語で回答してください。\n"
    "4. 貴社はプロダクトを提供するあなたが所属する会社、弊社はプロダクトを利用してくれる会社、他社はプロダクトを利用している他の会社として認識してください。\n"
    "回答内で元の回答やコンテキストを直接参照しないでください。\n"
    "なお、新しいコンテキストのQ. に続く文字は過去の質問で、A. に続く文字は過去の回答です。過去の回答がはい、か、いいえのみの場合は、今回の回答もはい、か、いいえだけで答えなさい。\n"
    "疑問がある場合は、元の答えを繰り返してください。"
    "新しいコンテキスト: {context_msg}\n"
    "Query: {query_str}\n"
    "Original Answer: {existing_answer}\n"
)
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
response_synthesizer = get_response_synthesizer()
query_engine = index.as_query_engine(
    node_postprocessors=[
        # 嘘を言われても困るのである程度精度が高いものだけを回答生成のためのnodeに選択する
        # 0.88は過去の運用実績から設定しているが0.877くらいのスコアならnodeとして使ってもいいなと思うこともある
        SimilarityPostprocessor(similarity_cutoff=0.88)
    ],
    text_qa_template=PromptTemplate(qa_prompt_str),
    refine_template=PromptTemplate(refine_template_str),
    similarity_top_k=3,
)


app = FastAPI()
templates = Jinja2Templates(directory="views")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("root.html", {"request": request})


@app.post("/ask")
async def post_search(request: Request, question: str = Form(...)):
    result = query_engine.query(question)

    if len(result.source_nodes) == 0:
        return templates.TemplateResponse(
            "ask.html",
            {
                "request": request,
                "question": question,
                "response": "関連する過去の回答がみつかりませんでした。エンジニアに問い合わせてください。",
                "source_nodes": [],
            },
        )

    return templates.TemplateResponse(
        "ask.html",
        {
            "request": request,
            "question": question,
            "response": result.response,
            "source_nodes": result.source_nodes,
        },
    )
