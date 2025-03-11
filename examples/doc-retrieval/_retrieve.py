import dsp
import dspy

# This is a fix to DSPy's error in retrieving with metadata
def _retrieve_with_metadata(queries, k, by_prob = True, **kwargs):
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """

    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if dsp.settings.reranker:
        return dsp.search.retrieveRerankEnsemblewithMetadata(queries=queries,k=k)

    queries = [q for q in queries if q]

    if len(queries) == 1:
        return dsp.search.retrievewithMetadata(queries[0], k)
    all_queries_passages = []
    for q in queries:
        passages = {}
        retrieved_passages = dsp.settings.rm(q, k=k * 3, **kwargs)
        for idx, psg in enumerate(retrieved_passages):
            if by_prob:
                passages[(idx, psg.long_text)] = (
                    passages.get(psg.long_text, 0.0) + psg.prob
                )
            else:
                passages[(idx, psg.long_text)] = (
                    passages.get(psg.long_text, 0.0) + psg.score
                )
            retrieved_passages[idx]["tracking_idx"] = idx
        passages = sorted(passages.items(), key=lambda item: item[1])[:k]
        req_indices = [psg[0][0] for psg in passages]
        passages = [
            rp for rp in retrieved_passages if rp.get("tracking_idx") in req_indices
        ]
        all_queries_passages.append(passages)
    return all_queries_passages

class _Retrieve(dspy.Retrieve):
    def forward(
        self,
        query_or_queries = None,
        query = None,
        k = None,
        by_prob = True,
        with_metadata = False,
        **kwargs,
    ):
        query_or_queries = query_or_queries or query

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [query.strip().split("\n")[0].strip() for query in queries]

        k = k if k is not None else self.k
        if not with_metadata:
            passages = dsp.retrieveEnsemble(queries, k=k, by_prob=by_prob, **kwargs)
            return dspy.Prediction(passages=passages)
        else:
            passages = _retrieve_with_metadata(
                queries, k=k, by_prob=by_prob, **kwargs,
            )
            if isinstance(passages[0], list):
                pred_returns = []
                for query_passages in passages:
                    passages_dict = {
                        key: []
                        for key in list(query_passages[0].keys())
                        if key != "tracking_idx"
                    }
                    for psg in query_passages:
                        for key, value in psg.items():
                            if key == "tracking_idx":
                                continue
                            passages_dict[key].append(value)
                    if "long_text" in passages_dict:
                        passages_dict["passages"] = passages_dict.pop("long_text")
                    pred_returns.append(dspy.Prediction(**passages_dict))
                return pred_returns
            elif isinstance(passages[0], dict):
                return dspy.retrieve.single_query_passage(passages=passages)