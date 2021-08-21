import pymongo
import dataclasses

from typing import List, Any

import torch
import transformers
import numpy as np

import json
import time
import os
import re


@dataclasses.dataclass
class HyperlinkedDocument:
    """
    A hyperlinked document contains text and a list of links, where each link has
        (a) anchor text in the original document as the source of the link
        (b) a link target that is typically a document ID or URL

    After tokenization, the document is represented with this class.

    For storing in a mmap, the class is serialized as two lists:
        (len(token_ids), num_links)
        a variable length list that combines the token ids, start/end indices and link target ids
    """

    token_ids: List[int]  # full document/untruncated tokenized text with special start/end tokens
    # The links are represented as these three lists of identical length.
    # They specify the link target and the start/end indices of the anchor text,
    # such that the anchor text for link k is token_ids[link_start_index[k]:link_end_index[k]]
    # Empty anchor text is not allowed, so that link_end_index[k] > link_start_index[k].  This is enforced in __init__ by dropping bad links.
    link_start_index: List[int]
    link_end_index: List[int]
    link_target: List[int]  # the link target id

    def __post_init__(self):
        # Remove empty anchor text
        link_start = []
        link_end = []
        link_target = []
        for start, end, target in zip(self.link_start_index, self.link_end_index, self.link_target):
            if start >= 0 and end > start:
                link_start.append(start)
                link_end.append(end)
                link_target.append(target)
        self.link_start_index = link_start
        self.link_end_index = link_end
        self.link_target = link_target

    def serialize(self):
        return (
            (len(self.token_ids), len(self.link_start_index)),
            self.token_ids + self.link_start_index + self.link_end_index + self.link_target,
        )

    @classmethod
    def deserialize(cls, num_token_ids_links, data):
        num_tokens, num_links = num_token_ids_links
        token_ids = data[:num_tokens]
        if num_links > 0:
            start_indices = data[num_tokens : (num_tokens + num_links)]
            end_indices = data[(num_tokens + num_links) : (num_tokens + 2 * num_links)]
            link_targets = data[-num_links:]
        else:
            start_indices = []
            end_indices = []
            link_targets = []
        assert len(start_indices) == len(end_indices) and len(end_indices) == len(link_targets)
        return cls(token_ids, start_indices, end_indices, link_targets)

    def to_string(self, tokenizer, index=None):
        """
        Use the tokenizer and index to convert ids to strings
        If index is provided then fetches the page target title, otherwise just decode the document and anchor text.
        """
        text = tokenizer.decode(self.token_ids)
        num_links = len(self.link_start_index)
        anchor_text = [
            tokenizer.decode(self.token_ids[self.link_start_index[k] : self.link_end_index[k]]) for k in range(num_links)
        ]
        if index is not None:
            link_target_titles = [index.id_to_title[self.link_target[k]] for k in range(num_links)]
        else:
            link_target_titles = None
        return {
            "text": text,
            "anchor_text": anchor_text,
            "link_targets": link_target_titles,
        }


def markup_wikipedia_page(page, special_token_start, special_token_end, get_links=True, index=None):
    # given the dictionary from the page, process it by marking the start/end of each link in the raw text, and combine the text from all sections.
    #
    # If get_links is True, then returns:
    #   text = a string with each anchor text surrounded by special_token_start, special_token_end
    #   a list of entities, each is a target page title
    #
    # If get_links is False, then just returns the text without surrounding the anchor text.
    #
    # If an index of links is available and provided, then it is used to resolve redirects and remove dead links.
    def _process_quotes_in_anchor_text(sentence, anchor_text):
        if "formatting" in sentence and "'" in anchor_text:
            for italic in sentence["formatting"].get("italic", []):
                anchor_text = re.sub(f"''({italic})''", r"\1", anchor_text)
            for bold in sentence["formatting"].get("bold", []):
                anchor_text = re.sub(f"'''({bold})'''", r"\1", anchor_text)
        return anchor_text.strip("'")

    sentences = []
    links = []

    sections_to_ignore = set(["External links", "References", "Further reading", "Notes", "See also"])

    for section in page["sections"]:
        if section.get("title") in sections_to_ignore:
            continue

        for paragraph in section.get("paragraphs", []):
            for sentence in paragraph["sentences"]:
                # We need to surround each link with the start/end tokens.
                # This requires finding each anchor text in the original text.
                # The links are ordered in the sentence, so we'll process it incrementally.

                text = " ".join(sentence["text"].strip().split())  # str
                sentence_text = []

                if get_links:
                    for link in sentence.get("links", []):
                        if link.get("type") != "internal":
                            # only get internal links for now
                            continue

                        target_id = link["page"]
                        if index is not None:
                            target_id = index.resolve_redirects(target_id)
                        if target_id is None:
                            # can't resolve this link target
                            continue

                        # get the anchor text
                        if link["text"] is None:
                            anchor_text = link["page"]
                        else:
                            anchor_text = link["text"]

                        anchor_text = " ".join(anchor_text.strip().split())
                        start_index = text.find(anchor_text)
                        if start_index == -1:
                            # need to deal with special formatting
                            try:
                                anchor_text = _process_quotes_in_anchor_text(sentence, anchor_text)
                            except:
                                # the regex didn't compile, skip it
                                continue
                            start_index = text.find(anchor_text)
                        if start_index == -1:
                            # print("Couldn't find anchor text")
                            # print(sentence)
                            # print(anchor_text)
                            continue  # skip it
                        end_index = start_index + len(anchor_text)

                        links.append(target_id)
                        sentence_text.extend(
                            [
                                text[:start_index],
                                special_token_start,
                                text[start_index:end_index],
                                special_token_end,
                            ]
                        )
                        text = text[end_index:]

                if len(text) > 0:
                    sentence_text.append(text)

                sentences.append("".join(sentence_text))

    document_text = " ".join(sentences)

    return document_text, links


def build_index_of_wikipedia_page_titles_and_redirects(outdir, min_page_length_in_characters=500):
    # build an index of:
    # (1) redirect source -> target
    # (2) page title -> page id as int (for kept pages longer then some length threshold)

    client = pymongo.MongoClient("localhost", 27017)
    db = client.enwiki
    pages = db.pages

    redirects = {}  # from source title to target title
    index = {}
    t1 = time.time()

    for page in pages.find():  # will get everything
        if not isinstance(page["_id"], str):
            # this isn't an ordinary page or redirect, skip it
            continue

        if page.get("isRedirect", False):
            if page["redirectTo"] is not None and page["redirectTo"].get("page") is not None:
                if not isinstance(page["redirectTo"]["page"], str):
                    continue
                redirects[page["_id"]] = page["redirectTo"]["page"]
        else:
            page_text, _ = markup_wikipedia_page(page, "", "", get_links=False)
            if len(page_text) > min_page_length_in_characters:
                index[page["_id"]] = len(index)

        if (len(index) + len(redirects)) % 1000 == 0:
            print("Finished {} pages, time {}".format(len(index) + len(redirects), time.time() - t1))

    with open(outdir + "/wikipedia_redirects.json", "w") as fout:
        json.dump(redirects, fout)

    with open(outdir + "/wikipedia_index.json", "w") as fout:
        json.dump(index, fout)


class WikipediaIndex:
    """
    Holds an index of page title -> int id and of source/target redirects
    """

    def __init__(self, index_dir):
        # Dict[str, str] of source --> target
        with open(os.path.join(index_dir, "wikipedia_redirects.json"), "r") as fin:
            self._source_to_target_redirect = json.load(fin)

        # Dict[str, int]
        with open(os.path.join(index_dir, "wikipedia_index.json"), "r") as fin:
            self._title_to_id = json.load(fin)

        self.id_to_title = {v: k for k, v in self._title_to_id.items()}

        self._stack_depth = 0  # NOTE: not thread safe
        self._maximum_redirects = 100

    def __len__(self):
        return len(self._title_to_id)

    def get_id_from_title(self, page_title: str) -> int:
        """
        Primary interface to look up an id from the page title as it also handles some custom wikipedia logic.
        """
        canonical_title = self.resolve_redirects(page_title)
        return self._title_to_id[canonical_title]

    def resolve_redirects(self, page_title: str, _reset_stack: bool = True):
        """
        Given a page title, resolve any redirects to return the final page title in the index.
        If the page isn't in the index, return None.  This function also canonicalizes the page_title with wikipedia specific processing.
        """
        if _reset_stack:
            self._stack_depth = 0
        else:
            self._stack_depth += 1
            if self._stack_depth > self._maximum_redirects:
                return None

        # need to check both page_title and page_title[0].upper() + page_title[1:]
        titles = [page_title]
        if len(page_title) > 0:
            titles.append(page_title[0].upper() + page_title[1:])

        # first check whether the title is in the index
        for title in titles:
            if title in self._title_to_id:
                return title

        # now check whether it is in the list of redirects
        for title in titles:
            if title in self._source_to_target_redirect:
                target_title = self._source_to_target_redirect[title]
                try:
                    ret = self.resolve_redirects(target_title, _reset_stack=False)
                except RecursionError:
                    ret = None
                if ret is not None:
                    return ret

        # Can't find it in the index or list of redirects
        return None


# a helper function
def tokenize_and_get_anchor_indices(text: str, tokenizer: Any, start_id: int, end_id: int):
    """
    Given some text with special begin anchor text/end anchor text tokens added surrounding the links,
    this function tokenizes the text and then processes the anchor text to:
        - remove the special token ids
        - produce a list of start/end indices for each link such that the anchor text can be reconstructed from the start/end indices
    """
    raw_token_ids = tokenizer.encode(text)
    token_ids = []
    start_indices = []
    end_indices = []

    next_index = 0
    for token_id in raw_token_ids:
        if token_id == start_id:
            start_indices.append(next_index)
        elif token_id == end_id:
            end_indices.append(next_index)
        else:
            token_ids.append(token_id)
            next_index += 1

    return token_ids, start_indices, end_indices


class HyperlinkedWikipediaMongo:
    """
    This class provides an interface to wikipedia stored on MongoDB to get documents in the HyperlinkedDocument format
    """

    def __init__(self, index_dir, tokenizer_name, hosthame="localhost", port=27017):
        """
        index_dir contains the pre-processed index of redirects and page title -> page int id
        """
        self._client = pymongo.MongoClient(hosthame, port)
        self._db = self._client.enwiki
        self._pages = self._db.pages

        self.index = WikipediaIndex(index_dir)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        if isinstance(self.tokenizer, transformers.models.t5.tokenization_t5_fast.T5TokenizerFast):
            self._special_token_start = "<extra_id_88>"
            self._special_token_end = "<extra_id_89>"
        else:
            raise ValueError("Unknown tokenizer")

        assert len(self.tokenizer.encode(self._special_token_start, add_special_tokens=False)) == 1
        self._special_token_start_id = self.tokenizer.encode(self._special_token_start, add_special_tokens=False)[0]
        assert len(self.tokenizer.encode(self._special_token_end, add_special_tokens=False)) == 1
        self._special_token_end_id = self.tokenizer.encode(self._special_token_end, add_special_tokens=False)[0]

    def get_page_from_mongo(self, page_title: str):
        # returns None if the page title isn't found
        sample = self._pages.find({"_id": page_title})

        page = None
        for page in sample:
            break

        return page

    def __len__(self):
        return len(self.index)

    def __getitem__(self, page_id: int):
        # Process:
        #   lookup the page title
        #   get the info from mongo using the page title
        #   get the text and links, and for each link resolve the redirects to get the target page title
        #   tokenize the text, replace the special tokens with start/end indices, map page title to page id

        # lookup the page title
        page_title = self.index.id_to_title[page_id]

        # get page from mongo
        page = self.get_page_from_mongo(page_title)

        # get the text and links, while resolving redirects for each link
        text, links = markup_wikipedia_page(page, self._special_token_start, self._special_token_end, index=self.index)

        # tokenize and get start/end indices for each anchor text
        token_ids, start_indices, end_indices = tokenize_and_get_anchor_indices(
            text,
            self.tokenizer,
            self._special_token_start_id,
            self._special_token_end_id,
        )

        # map the target page titles to ids
        link_target_ids = [self.index.get_id_from_title(link) for link in links]

        return HyperlinkedDocument(token_ids, start_indices, end_indices, link_target_ids)


class HyperlinkedWikipediaMMap:
    """
    This class provides an interface to wikipedia linked documents stored in a mmap

    The data is stored as two files:
        mmap_prefix + _lengths.npy = (num_documents * 3, ) array. Each document stores:
            - index of document start in data file
            - number of token_ids in the document
            - number of links in the document

        mmap_prefix + _data.npy = the token_ids, start/end indices and link targets for each document
    """

    def __init__(self, mmap_prefix):
        self._lengths = np.memmap(mmap_prefix + "_lengths.npy", mode="r", dtype="uint32")
        self.num_docs = len(self._lengths) // 3

        self._data = np.memmap(mmap_prefix + "_data.npy", mode="r", dtype="uint32")

    def __len__(self):
        return self.num_docs

    def __getitem__(self, page_id):
        start_index, num_tokens, num_links = self._lengths[
            (page_id * 3) : ((page_id + 1) * 3)
        ].tolist()  # tolist casts to python int from numpy.uint32
        end_index = start_index + num_tokens + 3 * num_links
        return HyperlinkedDocument.deserialize((num_tokens, num_links), self._data[start_index:end_index].tolist())

    def get_document_lengths(self):
        """
        Return a list of the number of tokens in each document
        """
        return self._lengths[1::3]


def build_wikipedia_mmap(outdir_prefix, index_dir, tokenizer_name):
    wiki_mongo = HyperlinkedWikipediaMongo(index_dir, tokenizer_name)

    lengths = []
    data = []

    t1 = time.time()
    for k in range(len(wiki_mongo)):
        doc = wiki_mongo[k]
        doc_lengths, doc_data = doc.serialize()
        doc_start_index = len(data)
        lengths.append(doc_start_index)
        lengths.extend(doc_lengths)
        data.extend(doc_data)
        if k % 1000 == 0:
            print(f"Finished {k} of {len(wiki_mongo)}, total time {time.time() - t1}")

    assert len(data) < 4294967295  # max size uint32

    # write out to mmap
    fp = np.memmap(outdir_prefix + "_lengths.npy", mode="w+", dtype="uint32", shape=(len(lengths),))
    fp[:] = np.array(lengths)[:]
    fp.flush()

    fp = np.memmap(outdir_prefix + "_data.npy", mode="w+", dtype="uint32", shape=(len(data),))
    fp[:] = np.array(data)[:]
    fp.flush()


if __name__ == "__main__":
    pass
