# Splice Tool Notes

Implementation history and worked examples for the Splice-facing tools
in `mcp_server/sample_engine/tools.py`. The tool docstrings carry the
operational contract (params, return shape, gating rules); this file
carries endpoint-capture history and full response examples.

## `get_splice_credits` — full response example (Ableton Live plan)

```json
{
  "connected": true,
  "username": "user-1367453956",
  "plan_raw": "subscribed",
  "plan_kind": "ableton_live",
  "sounds_plan_id": 12,
  "features": {"ableton_unmetered": true},
  "credits_remaining": 80,
  "credit_floor": 5,
  "daily_quota": {
    "used_today": 3, "remaining_today": 97, "daily_limit": 100,
    "near_limit": false, "at_limit": false
  },
  "can_download_sample": true,
  "download_gating": "daily_quota"
}
```

`download_gating` is `"daily_quota"` on the Ableton Live plan (the
100/day unmetered quota) or `"credit_floor"` on Sounds+/Creator/
Creator+ plans (the `CREDIT_HARD_FLOOR` protects the last 5 credits).
`connected: false` with zero credits means the Splice desktop app isn't
running or `grpcio` isn't installed.

## `splice_describe_sound` / `splice_generate_variation` — endpoint history

Both hit GraphQL operations on `surfaces-graphql.splice.com`, captured
via mitmproxy against Splice desktop 5.4.9 + the Sounds Plugin
(status: LIVE as of 2026-04-22):

- `splice_describe_sound` → `SamplesSearch` with `semantic=1` +
  `rephrase=true`. This is the Sounds Plugin's "Describe a Sound"
  feature — Splice's AI matches free-form descriptions like "dark
  ambient pad with shimmer" to catalog samples.
- `splice_generate_variation` → `AssetSimilarSoundsQuery` (the
  right-click "Variations" menu item). Up to 10 results, no credit
  cost — despite the tool name, this is a similarity-recommender
  lookup, not AI audio synthesis (the "generate" naming in the original
  handoff spec was aspirational).

Both tools strip the raw GraphQL response before returning (~270KB of
noise per call) — only the flattened sample list ships back.

`splice_search_with_sound` was removed 2026-04-22 — the user does this
in-Splice manually. The capture recipe for resurrecting it is at
`docs/2026-04-22-splice-https-capture-recipe.md`.
