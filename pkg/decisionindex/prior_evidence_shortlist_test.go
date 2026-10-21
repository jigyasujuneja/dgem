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
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ghchinoy/dgem/pkg/client"
	dgemtmpl "github.com/ghchinoy/dgem/pkg/template"
)

func TestFusePriorWithHesitationGate(t *testing.T) {
	optKeys := []string{"sku_a", "sku_b", "sku_c", "none_of_the_above"}

	// Case 1: Low raw hesitation (model is confident, rawNormH < 0.16) -> prior fusion is active.
	confidentRaw := map[string]float64{
		"sku_a":             0.96,
		"sku_b":             0.02,
		"sku_c":             0.01,
		"none_of_the_above": 0.01,
	}
	priors := map[string]float64{
		"sku_a":             2.0,
		"sku_b":             0.5,
		"sku_c":             0.0,
		"none_of_the_above": -1.0,
	}
	probs, winner, _, normH, rawH, rawNormH, fused := FusePriorWithHesitationGate(
		confidentRaw, priors, optKeys, 1.0, 0.50, 0.20,
	)
	if !fused {
		t.Fatalf("expected prior fusion to be active when rawNormH (%.4f) < hesitationGate (0.20)", rawNormH)
	}
	if winner != "sku_a" {
		t.Errorf("expected winner sku_a, got %q", winner)
	}
	if probs["sku_a"] <= confidentRaw["sku_a"] {
		t.Errorf("expected positive prior on sku_a to increase posterior probability: before=%.4f after=%.4f", confidentRaw["sku_a"], probs["sku_a"])
	}
	if rawH <= 0 || normH >= rawNormH {
		t.Errorf("expected post-fusion normH (%.4f) < rawNormH (%.4f)", normH, rawNormH)
	}

	// Case 2: High raw hesitation (model is unsure across sibling options, rawNormH >= 0.16) ->
	// even an extreme retrieval prior on sku_b must NOT fuse or mask the model's uncertainty!
	hesitantRaw := map[string]float64{
		"sku_a":             0.35,
		"sku_b":             0.33,
		"sku_c":             0.16,
		"none_of_the_above": 0.16,
	}
	extremeWrongPrior := map[string]float64{
		"sku_a":             -5.0,
		"sku_b":             10.0,
		"sku_c":             -5.0,
		"none_of_the_above": -5.0,
	}
	probs2, winner2, h2, normH2, rawH2, rawNormH2, fused2 := FusePriorWithHesitationGate(
		hesitantRaw, extremeWrongPrior, optKeys, 1.0, 0.80, 0.16,
	)
	if fused2 {
		t.Fatalf("expected Pre-Fusion Hesitation Gate to suppress prior fusion when rawNormH=%.4f >= 0.16", rawNormH2)
	}
	if winner2 != "sku_a" {
		t.Errorf("expected un-fused winner to remain sku_a, got %q", winner2)
	}
	if math.Abs(h2-rawH2) > 1e-9 || math.Abs(normH2-rawNormH2) > 1e-9 {
		t.Errorf("expected suppressed fusion to preserve exact raw entropy (h=%.4f vs rawH=%.4f)", h2, rawH2)
	}
	if probs2["sku_b"] > 0.40 {
		t.Errorf("expected suppressed fusion not to boost sku_b, got %.4f", probs2["sku_b"])
	}
}

func TestExecuteSystemOne_EvidenceSlotAndShortlist(t *testing.T) {
	var seenQuestionIDs []string
	var seenDescriptions map[string]string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var sysPayload struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					Name        string `json:"name"`
					Description string `json:"description"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}

		seenQuestionIDs = nil
		seenDescriptions = make(map[string]string)
		answers := make(map[string]client.QuestionAnswer)
		for _, q := range sysPayload.Questions {
			seenQuestionIDs = append(seenQuestionIDs, q.ID)
			for _, o := range q.Options {
				seenDescriptions[o.Name] = o.Description
			}
			if strings.HasSuffix(q.ID, "_evidence") {
				answers[q.ID] = client.QuestionAnswer{
					Choice: "decisive_match_cue",
					Probabilities: map[string]float64{
						"decisive_match_cue":      0.94,
						"conflicting_variant_cue": 0.02,
						"occluded_or_missing_cue": 0.02,
						"out_of_scope_candidate":  0.02,
					},
				}
				continue
			}
			chosen := q.Options[0].Name
			for _, o := range q.Options {
				if strings.Contains(o.Name, "sku_042") {
					chosen = o.Name
					break
				}
			}
			probs := make(map[string]float64, len(q.Options))
			rem := 0.03 / float64(maxInt(1, len(q.Options)-1))
			for _, o := range q.Options {
				if o.Name == chosen {
					probs[o.Name] = 0.97
				} else {
					probs[o.Name] = rem
				}
			}
			answers[q.ID] = client.QuestionAnswer{
				Choice:        chosen,
				Probabilities: probs,
			}
		}

		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": string(envBytes)}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)

	// Build a 120-option catalog question with PriorLogits and EvidencePrompt.
	criteria := make(map[string]string, 121)
	priors := make(map[string]float64, 121)
	for i := 0; i < 120; i++ {
		k := fmt.Sprintf("sku_%03d", i)
		if i == 42 {
			criteria[k] = "Gentle Skin Cleanser 500ml Single Pump Bottle"
			priors[k] = 3.2
		} else if i < 6 {
			criteria[k] = fmt.Sprintf("Gentle Skin Cleanser %d00ml Pack Variant %d", i+1, i)
			priors[k] = 2.5 - float64(i)*0.2
		} else {
			criteria[k] = fmt.Sprintf("Catalog item description %d", i)
			priors[k] = -1.0
		}
	}
	criteria["none_of_the_above"] = "None of the listed catalog items match the query image"
	priors["none_of_the_above"] = 0.1

	req := SystemOneRequest{
		State: "Query image crop from shelf position B3.",
		Questions: map[string]SystemOneQuestion{
			"catalog_match": {
				Type:           "choice",
				Instructions:   "Select the matching SKU for the query crop.",
				Criteria:       criteria,
				PriorLogits:    priors,
				EvidencePrompt: "Identify whether the pack-size and variant badge on the query crop match a candidate.",
			},
		},
	}

	opts := DefaultEngineOptions()
	opts.ShortlistTopK = 6

	resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}

	// Verify 120-option question resolved in a SINGLE forward pass (1 pass instead of 7 tournament passes)!
	if resp.EvaluationTrace.ForwardPasses != 1 {
		t.Errorf("expected 1 forward pass via ShortlistFunc, got %d", resp.EvaluationTrace.ForwardPasses)
	}
	// Verify co-denoised evidence slot preceded catalog_match on the canvas.
	if len(seenQuestionIDs) != 2 || seenQuestionIDs[0] != "catalog_match_evidence" || seenQuestionIDs[1] != "catalog_match" {
		t.Errorf("expected canvas slot order [catalog_match_evidence, catalog_match], got %v", seenQuestionIDs)
	}
	// Verify contrastive criteria enrichment added [distinguishing: ...] to sku_042.
	if !strings.Contains(seenDescriptions["sku_042"], "[distinguishing:") {
		t.Errorf("expected BuildContrastiveCriteria to enrich sku_042 description, got %q", seenDescriptions["sku_042"])
	}

	ans := resp.Answers["catalog_match"]
	if ans.Choice != "sku_042" {
		t.Errorf("expected winner sku_042, got %q", ans.Choice)
	}
	if ans.Evidence != "decisive_match_cue" {
		t.Errorf("expected Evidence='decisive_match_cue', got %q", ans.Evidence)
	}
	if !ans.PriorFused {
		t.Errorf("expected PriorFused=true for low-hesitation readout")
	}
	if len(ans.Probabilities) != 121 {
		t.Errorf("expected probabilities over all 121 options, got %d", len(ans.Probabilities))
	}
	sumP := 0.0
	for _, p := range ans.Probabilities {
		if p <= 0 {
			t.Errorf("expected strictly positive probability for every option, got %v", p)
		}
		sumP += p
	}
	if math.Abs(sumP-1.0) > 1e-6 {
		t.Errorf("expected probabilities to sum to 1.0, got %.8f", sumP)
	}
}

func TestCheckTriangleConsistency(t *testing.T) {
	edges := []EntityPairEdge{
		// Inconsistent triangle (rec_1, rec_2, rec_3): 1==2 (0.91), 2==3 (0.88), but 1!=3 (0.12)
		{EntityA: "rec_1", EntityB: "rec_2", DuplicateProb: 0.91},
		{EntityA: "rec_2", EntityB: "rec_3", DuplicateProb: 0.88},
		{EntityA: "rec_1", EntityB: "rec_3", DuplicateProb: 0.12},
		// Consistent triangle (rec_4, rec_5, rec_6): all high
		{EntityA: "rec_4", EntityB: "rec_5", DuplicateProb: 0.93},
		{EntityA: "rec_5", EntityB: "rec_6", DuplicateProb: 0.89},
		{EntityA: "rec_4", EntityB: "rec_6", DuplicateProb: 0.85},
	}

	contradictions := CheckTriangleConsistency(edges, 0.80, 0.25)
	if len(contradictions) != 1 {
		t.Fatalf("expected 1 triangle contradiction, got %d: %+v", len(contradictions), contradictions)
	}
	c := contradictions[0]
	if c.Nodes != [3]string{"rec_1", "rec_2", "rec_3"} {
		t.Errorf("unexpected contradiction nodes: %v", c.Nodes)
	}
	if math.Abs(c.SeverityGap-(0.91-0.12)) > 1e-6 {
		t.Errorf("expected SeverityGap=0.79, got %.4f", c.SeverityGap)
	}
}

func TestNewTemplatesRenderValidJSON(t *testing.T) {
	eng := dgemtmpl.NewEngine()
	templates := []struct {
		relPath       string
		wantQuestions int
	}{
		{"../../templates/multimodal/contact_sheet_verify.json.tmpl", 4},
		{"../../templates/multimodal/ui_pathfinder_som.json.tmpl", 4},
		{"../../templates/multimodal/entity_resolution_dedup.json.tmpl", 4},
		{"../../templates/live_checklist_verify.json.tmpl", 8},
		{"../../templates/swarm_lock_arbiter.json.tmpl", 8},
	}

	for _, tc := range templates {
		rendered, err := eng.RenderFile(filepath.Clean(tc.relPath), nil)
		if err != nil {
			t.Fatalf("RenderFile(%s) failed: %v", tc.relPath, err)
		}
		var doc struct {
			Schema struct {
				Questions []map[string]any `json:"questions"`
			} `json:"schema"`
			State map[string]any `json:"state"`
		}
		if err := json.Unmarshal([]byte(rendered), &doc); err != nil {
			t.Fatalf("template %s rendered invalid JSON: %v\n%s", tc.relPath, err, rendered)
		}
		if len(doc.Schema.Questions) != tc.wantQuestions {
			t.Errorf("template %s: got %d questions, want %d", tc.relPath, len(doc.Schema.Questions), tc.wantQuestions)
		}
	}
}
