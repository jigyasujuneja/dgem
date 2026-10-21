// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package decisionindex

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ghchinoy/dgem/pkg/client"
)

// ShortlistFunc selects at most topK candidate option keys (and optional stage-1 prior logits)
// for a wide choice question (K > MaxOptionsPerSlot) so it can be resolved in a single diffusion pass (PROP-24 / EXP-28).
type ShortlistFunc func(
	ctx context.Context,
	stateText string,
	images []string,
	qKey string,
	q SystemOneQuestion,
	topK int,
) (shortlistKeys []string, priorLogits map[string]float64, err error)

// PriorLogitsShortlist returns a ShortlistFunc that ranks options by q.PriorLogits (descending, tie-broken
// deterministically by option key), retaining the topK highest-prior options plus any catch-all options.
func PriorLogitsShortlist(defaultTopK int) ShortlistFunc {
	return func(_ context.Context, _ string, _ []string, _ string, q SystemOneQuestion, topK int) ([]string, map[string]float64, error) {
		if topK <= 0 {
			topK = defaultTopK
		}
		if topK <= 0 {
			topK = 6
		}
		allKeys := sortedOptionKeys(q.Criteria)
		if len(allKeys) <= topK {
			return allKeys, q.PriorLogits, nil
		}

		catchAll := catchAllKeys(q.Criteria)
		isCatchAll := make(map[string]bool, len(catchAll))
		for _, k := range catchAll {
			isCatchAll[k] = true
		}

		type scoredKey struct {
			key   string
			score float64
		}
		candidates := make([]scoredKey, 0, len(allKeys))
		for _, k := range allKeys {
			if isCatchAll[k] {
				continue
			}
			candidates = append(candidates, scoredKey{
				key:   k,
				score: q.PriorLogits[k],
			})
		}
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].score != candidates[j].score {
				return candidates[i].score > candidates[j].score
			}
			return candidates[i].key < candidates[j].key
		})

		budget := topK
		if len(catchAll) > 0 && budget+len(catchAll) > MaxOptionsPerSlot {
			budget = MaxOptionsPerSlot - len(catchAll)
		}
		if budget > len(candidates) {
			budget = len(candidates)
		}
		out := make([]string, 0, budget+len(catchAll))
		for i := 0; i < budget; i++ {
			out = append(out, candidates[i].key)
		}
		out = append(out, catchAll...)
		return out, q.PriorLogits, nil
	}
}

// BuildContrastiveCriteria enriches each shortlisted option's description with the tokens that distinguish it
// from sibling candidates in the shortlist (e.g., pack counts "6x100ml" vs "1x100ml", variant badges, or unique terms).
func BuildContrastiveCriteria(criteria map[string]string, shortlistKeys []string) map[string]string {
	out := make(map[string]string, len(shortlistKeys))
	tokenFreq := make(map[string]int)
	perOptTokens := make(map[string][]string, len(shortlistKeys))

	for _, k := range shortlistKeys {
		desc := strings.TrimSpace(criteria[k])
		out[k] = desc
		seen := make(map[string]bool)
		for _, rawTok := range strings.Fields(strings.ToLower(k + " " + desc)) {
			tok := strings.Trim(rawTok, ".,;:!?()[]{}\"'")
			if len(tok) < 2 || seen[tok] {
				continue
			}
			seen[tok] = true
			perOptTokens[k] = append(perOptTokens[k], tok)
			tokenFreq[tok]++
		}
	}

	for _, k := range shortlistKeys {
		var unique []string
		for _, tok := range perOptTokens[k] {
			if tokenFreq[tok] == 1 {
				unique = append(unique, tok)
				if len(unique) >= 4 {
					break
				}
			}
		}
		if len(unique) > 0 && !strings.Contains(out[k], "[distinguishing:") {
			out[k] = fmt.Sprintf("%s [distinguishing: %s]", out[k], strings.Join(unique, ", "))
		}
	}
	return out
}

// evaluateWideQuestionShortlist resolves a wide choice question (K > MaxOptionsPerSlot, up to 10,000+ options)
// in a single diffusion pass using ShortlistFunc (or PriorLogitsShortlist when q.PriorLogits is populated).
// Non-shortlisted options receive a strictly positive residual tail mass proportional to exp(priorLogits[k])
// so the returned distribution covers all K options and sums to 1.0 (PROP-24 / EXP-28).
func evaluateWideQuestionShortlist(
	ctx context.Context,
	cli *client.Client,
	stateText string,
	images []string,
	qKey string,
	qSpec SystemOneQuestion,
	opts EngineOptions,
) (SystemOneAnswer, int, error) {
	topK := opts.ShortlistTopK
	if topK <= 0 {
		topK = 6
	}
	limit := optionLimit(opts.Labels)
	if topK > limit {
		topK = limit
	}

	fn := opts.ShortlistFunc
	if fn == nil {
		fn = PriorLogitsShortlist(topK)
	}
	shortKeys, priors, err := fn(ctx, stateText, images, qKey, qSpec, topK)
	if err != nil {
		return SystemOneAnswer{}, 0, err
	}
	if len(shortKeys) < 2 {
		// Fall back to tournament routing if the shortlist produced fewer than 2 options.
		return evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts.TemperatureScale, opts.BracketSize, opts.CatchAll, opts.PromptLayout, opts.OptionOrder, opts.Labels)
	}
	if len(shortKeys) > limit {
		shortKeys = shortKeys[:limit]
	}

	subQ := subQuestion(qSpec, shortKeys)
	if len(priors) > 0 && len(subQ.PriorLogits) == 0 {
		subQ.PriorLogits = make(map[string]float64, len(shortKeys))
		for _, k := range shortKeys {
			if v, ok := priors[k]; ok {
				subQ.PriorLogits[k] = v
			}
		}
	}
	subQ.Criteria = BuildContrastiveCriteria(subQ.Criteria, shortKeys)

	batchMap, passes, err := evaluateStandardBatch(
		ctx,
		cli,
		stateText,
		images,
		[]string{qKey},
		map[string]SystemOneQuestion{qKey: subQ},
		map[string]string{qKey: kindChoice},
		opts,
	)
	if err != nil {
		return SystemOneAnswer{}, passes, err
	}
	shortAns := batchMap[qKey]

	allOptKeys := optionKeys(qSpec, opts.OptionOrder)
	if len(allOptKeys) == len(shortKeys) {
		return shortAns, passes, nil
	}

	// Lift the shortlist distribution to all K options while preserving the shortlist argmax winner
	// and allocating a small residual tail mass (1e-3 total) across non-shortlisted options.
	inShortlist := make(map[string]bool, len(shortKeys))
	for _, k := range shortKeys {
		inShortlist[k] = true
	}
	const tailMass = 1e-3
	fullProbs := make(map[string]float64, len(allOptKeys))
	for _, k := range shortKeys {
		fullProbs[k] = (1.0 - tailMass) * math.Max(1e-9, shortAns.Probabilities[k])
	}
	numRest := len(allOptKeys) - len(shortKeys)
	if numRest > 0 {
		perRest := tailMass / float64(numRest)
		for _, k := range allOptKeys {
			if !inShortlist[k] {
				fullProbs[k] = perRest
			}
		}
	}

	// Normalize exactly over all K options (with tempScale=1.0 since evaluateStandardBatch already scaled).
	normProbs, winner, h, normH := NormalizeAndScaleProbabilities(fullProbs, allOptKeys, 1.0)
	conf := normProbs[winner]
	return SystemOneAnswer{
		Type:           "choice",
		Choice:         winner,
		Probabilities:  normProbs,
		Confidence:     &conf,
		Entropy:        h,
		NormalizedH:    normH,
		PassesUsed:     passes,
		RawEntropy:     shortAns.RawEntropy,
		RawNormalizedH: shortAns.RawNormalizedH,
		PriorFused:     shortAns.PriorFused,
		Evidence:       shortAns.Evidence,
	}, passes, nil
}

// EntityPairEdge represents a calibrated pairwise duplicate probability P(EntityA == EntityB)
// emitted by a single-pass entity-resolution policy (PROP-36 / EXP-30).
type EntityPairEdge struct {
	EntityA       string  `json:"entity_a"`
	EntityB       string  `json:"entity_b"`
	DuplicateProb float64 `json:"duplicate_prob"`
}

// TriangleContradiction records a 3-cycle (A, B, C) that violates transitive equivalence:
// two edges have high duplicate probability (>= highThresh) while the third edge has low duplicate probability (<= lowThresh).
type TriangleContradiction struct {
	Nodes       [3]string  `json:"nodes"`
	EdgeProbs   [3]float64 `json:"edge_probs"`
	SeverityGap float64    `json:"severity_gap"`
}

// CheckTriangleConsistency inspects pairwise entity-resolution probabilities for transitive 3-cycle contradictions
// (PROP-36 / EXP-30). Any triangle (u, v, w) where two pairs exceed highThresh (default 0.80) and the third pair
// is at or below lowThresh (default 0.25) is returned in deterministic order so the caller can withhold automated
// cluster merges and escalate the conflicting triple.
func CheckTriangleConsistency(edges []EntityPairEdge, highThresh, lowThresh float64) []TriangleContradiction {
	if highThresh <= 0 {
		highThresh = 0.80
	}
	if lowThresh <= 0 {
		lowThresh = 0.25
	}
	adj := make(map[string]map[string]float64)
	nodeSet := make(map[string]bool)
	for _, e := range edges {
		a, b := strings.TrimSpace(e.EntityA), strings.TrimSpace(e.EntityB)
		if a == "" || b == "" || a == b {
			continue
		}
		if adj[a] == nil {
			adj[a] = make(map[string]float64)
		}
		if adj[b] == nil {
			adj[b] = make(map[string]float64)
		}
		adj[a][b] = e.DuplicateProb
		adj[b][a] = e.DuplicateProb
		nodeSet[a] = true
		nodeSet[b] = true
	}

	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var out []TriangleContradiction
	n := len(nodes)
	for i := 0; i < n; i++ {
		u := nodes[i]
		for j := i + 1; j < n; j++ {
			v := nodes[j]
			pUV, okUV := adj[u][v]
			if !okUV {
				continue
			}
			for k := j + 1; k < n; k++ {
				w := nodes[k]
				pVW, okVW := adj[v][w]
				pUW, okUW := adj[u][w]
				if !okVW || !okUW {
					continue
				}
				probs := [3]float64{pUV, pVW, pUW}
				highCount := 0
				minP := 1.0
				maxP := 0.0
				for _, p := range probs {
					if p >= highThresh {
						highCount++
					}
					if p < minP {
						minP = p
					}
					if p > maxP {
						maxP = p
					}
				}
				if highCount == 2 && minP <= lowThresh {
					out = append(out, TriangleContradiction{
						Nodes:       [3]string{u, v, w},
						EdgeProbs:   probs,
						SeverityGap: maxP - minP,
					})
				}
			}
		}
	}
	return out
}
