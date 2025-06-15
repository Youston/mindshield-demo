from mindshield_core.retrieval import WindowRetriever

class DummyIndex(list):
    def get(self, _id):
        for doc in self:
            if doc["id"] == _id:
                return doc
        return None

def test_find_hits():
    idx = DummyIndex([
        {"id": "1", "text": "This is a test document about anxiety and panic attacks."}
    ])
    retriever = WindowRetriever(idx, window_size=50, margin=5)
    spans = retriever.find_hits(idx[0]["text"], ["anxiety"])
    assert spans, "Should find at least one span" 