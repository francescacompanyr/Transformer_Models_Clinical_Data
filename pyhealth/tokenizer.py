from typing import List, Optional, Tuple


class Vocabulary:

    def __init__(self, tokens: List[str]):

        all_tokens =  tokens
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        for token in all_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def __call__(self, token):
        """Retrieves the index of the token.Note that if the token is not in the vocabulary, this function will try to
        return the index of <unk>. If <unk> is not in the vocabulary, an exception will be raised.
        """
        return self.token2idx[token]

    def __len__(self):
        return len(self.token2idx)

    def __contains__(self, token):
        return token in self.token2idx


class Tokenizer:

    def __init__(self, tokens: List[str]):
        self.vocabulary = Vocabulary(tokens=tokens)


    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def convert_tokens_to_indices(self, tokens: List[str]) -> List[int]:
        return [self.vocabulary(token) for token in tokens]

    def convert_indices_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.vocabulary.idx2token[idx] for idx in indices]

    def batch_encode_2d(
        self,
        batch: List[List[str]],
        max_length: int = 512, ):

        return [[self.vocabulary(token) for token in tokens] for tokens in batch]

    def batch_decode_2d(self, batch: List[List[int]], padding: bool = False, ):
        batch = [[self.vocabulary.idx2token[idx] for idx in tokens] for tokens in batch]

        return batch

    def batch_encode_3d(
        self,
        batch: List[List[List[str]]],
        padding: Tuple[bool, bool] = (True, True),
        truncation: Tuple[bool, bool] = (True, True),
        max_length: Tuple[int, int] = (10, 512), ):
       

        return [[[self.vocabulary(token) for token in tokens] for tokens in visits]
            for visits in batch]

    def batch_decode_3d(
        self, batch: List[List[List[int]]],
        padding: bool = False,):
        batch = [ self.batch_decode_2d(batch=visits, padding=padding) for visits in batch ]
        if not padding:
            batch = [[visit for visit in visits if visit != []] for visits in batch]
        return batch


