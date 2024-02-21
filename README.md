# Multi-granularity repurchase interval-aware recommendation with small item-sets

Understanding users' repurchase patterns is crucial for improving the quality of item recommendations. Many studies have straightforwardly modeled the intervals between users' repeat purchases using a single distribution. However, the empirical distribution of repurchase intervals for certain products on e-commerce platforms often resembles a mixture distribution. Typically, this mixture consists of one major unimodal distribution coupled with multiple Gaussian distributions of different time granularities. Currently available recommendation systems cannot achieve optimal effectiveness if this fact is not accounted for, especially when recommending with a small item-set. Based on this finding, we propose MgRIA, a BERT-based recommendation model with a user repeat purchase feature block. This module includes a flexible scoring mechanism designed to accommodate data following a mixture distribution, as observed in repeat purchase behaviors. We applied MgRIA and several existing recommendation methods to various datasets and found that MgRIA achieves the best overall performance.

recommendation system; repurchase time interval; multi-granularity; mixture distribution; attention mechanism

codes are to be released soon
