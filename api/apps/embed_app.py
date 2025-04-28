# -*- coding: utf-8 -*-
# 2025/4/28
# create by: snower

import json
import logging
from copy import deepcopy

import numpy as np
from flask import Response, request
from flask_login import current_user, login_required

from api import settings
from api.db import LLMType, ParserType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.utils.api_utils import validate_request
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts import chunks_format, kb_prompt
from rag.settings import PAGERANK_FLD
from rag.utils import rmSpace


@manager.route("/ask", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def embed_search_ask():
    req = request.json
    uid = current_user.id

    def stream():
        nonlocal req, uid
        try:
            for ans in do_ask(req["question"], req["kb_ids"], uid):
                yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp

def do_ask(question, kb_ids, tenant_id):
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
    retriever = settings.retrievaler if not is_knowledge_graph else settings.kg_retrievaler

    embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embedding_list[0])
    rerank_mdl = LLMBundle(tenant_id, LLMType.RERANK)
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
    max_tokens = chat_mdl.max_length
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))
    kbinfos = do_retrieval(retriever, question, embd_mdl, tenant_ids, kb_ids, 1, 100, 0.3, 0.5, aggs=False,
                                  rerank_mdl=rerank_mdl, rank_feature=label_question(question, kbs))
    logging.info("embed do_ask retrieval")
    knowledges = kb_prompt(kbinfos, max_tokens)
    logging.info("embed do_ask kb_prompt")
    prompt = """
    Role: You're a smart assistant. Your name is Miss R.
    Task: Summarize the information from knowledge bases and answer user's question.
    Requirements and restriction:
      - DO NOT make things up, especially for numbers.
      - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
      - Answer with markdown format text.
      - Answer in language of user's question.
      - DO NOT make things up, especially for numbers.

    ### Information from knowledge bases
    %s

    The above is information from knowledge bases.

    """ % "\n".join(knowledges)
    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, prompt
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]], [ck["vector"] for ck in kbinfos["chunks"]], embd_mdl, tkweight=0.7, vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        refs["chunks"] = chunks_format(refs)
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)

def do_retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
              vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
              rerank_mdl=None, highlight=False,
              rank_feature: dict | None = {PAGERANK_FLD: 10}):
    ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
    if not question:
        return ranks

    req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "page": 1, "size": page_size * 2,
           "question": question, "vector": True, "topk": top,
           "similarity": similarity_threshold,
           "available_int": 1}


    if isinstance(tenant_ids, str):
        tenant_ids = tenant_ids.split(",")

    sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                       kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
    logging.info("embed do_retrieval search")

    if rerank_mdl and sres.total > 0:
        sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                               sres, question, 1 - vector_similarity_weight,
                                               vector_similarity_weight,
                                               rank_feature=rank_feature)
        logging.info("embed do_retrieval rerank_mdl")
    else:
        sim, tsim, vsim = self.rerank(
            sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
            rank_feature=rank_feature)
    # Already paginated in search function
    idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]


    dim = len(sres.query_vector)
    vector_column = f"q_{dim}_vec"
    zero_vector = [0.0] * dim
    if doc_ids:
        similarity_threshold = 0
        page_size = 30
    sim_np = np.array(sim)
    filtered_count = (sim_np >= similarity_threshold).sum()
    ranks["total"] = int(filtered_count) # Convert from np.int64 to Python int otherwise JSON serializable error
    for i in idx:
        if sim[i] < similarity_threshold:
            break
        if len(ranks["chunks"]) >= page_size:
            if aggs:
                continue
            break
        id = sres.ids[i]
        chunk = sres.field[id]
        dnm = chunk.get("docnm_kwd", "")
        did = chunk.get("doc_id", "")
        position_int = chunk.get("position_int", [])
        d = {
            "chunk_id": id,
            "content_ltks": chunk["content_ltks"],
            "content_with_weight": chunk["content_with_weight"],
            "doc_id": did,
            "docnm_kwd": dnm,
            "kb_id": chunk["kb_id"],
            "important_kwd": chunk.get("important_kwd", []),
            "image_id": chunk.get("img_id", ""),
            "similarity": sim[i],
            "vector_similarity": vsim[i],
            "term_similarity": tsim[i],
            "vector": chunk.get(vector_column, zero_vector),
            "positions": position_int,
        }
        if highlight and sres.highlight:
            if id in sres.highlight:
                d["highlight"] = rmSpace(sres.highlight[id])
            else:
                d["highlight"] = d["content_with_weight"]
        ranks["chunks"].append(d)
        if dnm not in ranks["doc_aggs"]:
            ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
        ranks["doc_aggs"][dnm]["count"] += 1
    ranks["doc_aggs"] = [{"doc_name": k,
                          "doc_id": v["doc_id"],
                          "count": v["count"]} for k, v in sorted(ranks["doc_aggs"].items(),
                                                               key=lambda x: x[1]["count"] * -1)]
    ranks["chunks"] = ranks["chunks"][:page_size]

    return ranks