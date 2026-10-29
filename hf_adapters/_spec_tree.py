# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side token-tree helpers for tree speculative decoding (batch 1).

A tree is ``(tokens, parents)``: node 0 is the root (``parents[0] == -1``),
every other node extends its parent by one token, parents precede their
children, and siblings carry distinct tokens.
"""

import heapq


def tree_depths(parents):
    """Depth of every node of an ancestor-closed tree (``parents[0] == -1``)."""
    if not parents or parents[0] != -1:
        raise ValueError("node 0 must be the root (parent -1)")
    depths = [0]
    for i, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < i:
            raise ValueError(f"node {i}: parent {parent} must precede it")
        depths.append(depths[parent] + 1)
    return depths


def tree_ancestors(parents):
    """``ancestors[j]``: the nodes on the root-to-``j`` path, ``j`` included."""
    tree_depths(parents)
    paths = [[0]]
    for i, parent in enumerate(parents[1:], 1):
        paths.append(paths[parent] + [i])
    return paths


def accept_greedy(tokens, parents, predictions):
    """Longest tree path matching the target's greedy predictions.

    ``predictions[j]`` is the target's argmax after node ``j``. Returns
    ``(path, bonus)``: the accepted node indices, root first, and the
    target's token after the last accepted node (always emitted).
    """
    children = {}
    for i, parent in enumerate(parents[1:], 1):
        key = (parent, int(tokens[i]))
        if key in children:
            raise ValueError(f"duplicate sibling token {key[1]} under node {parent}")
        children[key] = i
    path = [0]
    while True:
        bonus = int(predictions[path[-1]])
        child = children.get((path[-1], bonus))
        if child is None:
            return path, bonus
        path.append(child)


def pad_tree(tokens, parents, width):
    """Fill a short tree to ``width`` nodes with extra root children.

    A verify graph has a fixed width. Filler children carry tokens unused
    among the root's children, so ``accept_greedy`` stays unique.
    """
    tokens, parents = list(tokens), list(parents)
    used = {t for t, p in zip(tokens, parents) if p == 0}
    filler = 0
    while len(tokens) < width:
        while filler in used:
            filler += 1
        tokens.append(filler)
        parents.append(0)
        used.add(filler)
    return tokens, parents


def best_first_tree(root, slot_ids, slot_logprobs, budget, max_depth=None):
    """The ``budget`` highest-scoring nodes of a tree of independent slots.

    ``slot_ids[d]`` / ``slot_logprobs[d]`` (``[S, K]``, sorted descending)
    are the top-``K`` candidates of the token at depth ``d + 1``, independent
    of the path (a block drafter such as DFlash). A node's score is the sum
    of its path's log-probs, so a child never outscores its parent and the
    result is ancestor-closed. Enumeration is exact and lazy: popping a node
    pushes only its next sibling and its first child.

    Returns ``(tokens, parents)`` with node 0 the root.
    """
    depth_limit = len(slot_ids) if max_depth is None else min(max_depth, len(slot_ids))
    width = len(slot_ids[0]) if depth_limit else 0
    tokens, parents, depths, scores = [int(root)], [-1], [0], [0.0]
    heap = []
    if depth_limit and width:
        heap.append((-float(slot_logprobs[0][0]), 0, 0, 0))
    counter = 1
    while heap and len(tokens) < budget:
        neg, _, parent, rank = heapq.heappop(heap)
        depth = depths[parent] + 1
        tokens.append(int(slot_ids[depth - 1][rank]))
        parents.append(parent)
        depths.append(depth)
        scores.append(-neg)
        node = len(tokens) - 1
        if rank + 1 < width:
            score = scores[parent] + float(slot_logprobs[depth - 1][rank + 1])
            heapq.heappush(heap, (-score, counter, parent, rank + 1))
            counter += 1
        if depth < depth_limit:
            score = scores[node] + float(slot_logprobs[depth][0])
            heapq.heappush(heap, (-score, counter, node, 0))
            counter += 1
    return tokens, parents
