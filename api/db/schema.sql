-- ============================================================
-- Salus Content Moderation API – Supabase Schema
-- Run this once in the Supabase SQL Editor to bootstrap the DB.
-- ============================================================


-- ------------------------------------------------------------
-- 1. RAW EVENTS TABLE
--    One row per (image, recognizer) pair processed by the API.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS detection_events (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      timestamptz NOT NULL DEFAULT now(),

    -- which model produced this result
    recognizer      text        NOT NULL
                    CHECK (recognizer IN ('deepfake', 'nsfw', 'flux')),

    -- which API route was called (/detect/image/deepfake, /detect/image, …)
    endpoint        text        NOT NULL,

    -- final decision
    is_flagged      boolean     NOT NULL,
    label           text        NOT NULL,
    score           float8      NOT NULL CHECK (score >= 0 AND score <= 1),

    -- full per-class probability list from the model
    all_predictions jsonb       NOT NULL DEFAULT '[]'::jsonb,

    -- de-duplication: SHA-256 hex of the raw image bytes (NULL for legacy rows)
    image_hash      text,

    -- how many times this (image, recognizer) pair has been submitted
    scan_count      int         NOT NULL DEFAULT 1,

    -- how many of those submissions resulted in a flag
    times_flagged   int         NOT NULL DEFAULT 0
);

-- ── Indexes ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS detection_events_created_at_idx
    ON detection_events (created_at DESC);

CREATE INDEX IF NOT EXISTS detection_events_recognizer_idx
    ON detection_events (recognizer);

CREATE INDEX IF NOT EXISTS detection_events_is_flagged_idx
    ON detection_events (is_flagged);

-- Composite index for recognizer-based time series queries.
-- We avoid date_trunc(created_at) in index expressions because
-- timestamptz truncation is not immutable in PostgreSQL.
CREATE INDEX IF NOT EXISTS detection_events_recognizer_created_at_idx
    ON detection_events (recognizer, created_at DESC);

-- Partial unique index used by the upsert ON CONFLICT clause.
-- NULL image_hash rows (legacy / hash-less inserts) are excluded.
CREATE UNIQUE INDEX IF NOT EXISTS detection_events_image_hash_recognizer_uidx
    ON detection_events (image_hash, recognizer)
    WHERE image_hash IS NOT NULL;


-- ------------------------------------------------------------
-- 2. ROW-LEVEL SECURITY
--    Service role (server-side writes) has full access.
--    Anonymous role is intentionally denied—stats endpoints
--    should be protected by your own auth layer.
-- ------------------------------------------------------------

ALTER TABLE detection_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_full_access"
    ON detection_events
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);


-- ------------------------------------------------------------
-- 3. VIEWS
-- ------------------------------------------------------------

-- 3a. All-time statistics per recognizer
CREATE OR REPLACE VIEW recognizer_overview AS
SELECT
    recognizer,
    COUNT(*)                                                            AS total_scans,
    SUM(is_flagged::int)                                               AS flagged_count,
    COUNT(*) - SUM(is_flagged::int)                                    AS clean_count,
    ROUND(
        100.0 * SUM(is_flagged::int) / NULLIF(COUNT(*), 0), 2
    )                                                                   AS flag_rate_pct,
    ROUND(AVG(score)::numeric, 4)                                      AS avg_score,
    ROUND(AVG(CASE WHEN is_flagged     THEN score END)::numeric, 4)   AS avg_flagged_score,
    ROUND(AVG(CASE WHEN NOT is_flagged THEN score END)::numeric, 4)   AS avg_clean_score,
    MIN(created_at)                                                     AS first_scan_at,
    MAX(created_at)                                                     AS last_scan_at
FROM detection_events
GROUP BY recognizer;


-- 3b. Daily breakdown per recognizer (used by trend charts)
CREATE OR REPLACE VIEW daily_detection_summary AS
SELECT
    date_trunc('day', created_at)                                      AS day,
    recognizer,
    COUNT(*)                                                            AS total_scans,
    SUM(is_flagged::int)                                               AS flagged_count,
    COUNT(*) - SUM(is_flagged::int)                                    AS clean_count,
    ROUND(
        100.0 * SUM(is_flagged::int) / NULLIF(COUNT(*), 0), 2
    )                                                                   AS flag_rate_pct,
    ROUND(AVG(score)::numeric, 4)                                      AS avg_score,
    ROUND(AVG(CASE WHEN is_flagged     THEN score END)::numeric, 4)   AS avg_flagged_score
FROM detection_events
GROUP BY 1, 2
ORDER BY 1 DESC, 2;


-- 3c. Hourly scan+flag counts for the last 7 days (rolling dashboard)
CREATE OR REPLACE VIEW recent_hourly_activity AS
SELECT
    date_trunc('hour', created_at)  AS hour,
    recognizer,
    COUNT(*)                         AS scans,
    SUM(is_flagged::int)             AS flagged
FROM detection_events
WHERE created_at >= now() - INTERVAL '7 days'
GROUP BY 1, 2
ORDER BY 1 DESC, 2;


-- 3d. Score distribution in 10 equal-width buckets per recognizer
--     bucket 1 = [0.0, 0.1), bucket 10 = [0.9, 1.0]
CREATE OR REPLACE VIEW score_distribution AS
SELECT
    recognizer,
    width_bucket(score, 0.0, 1.0001, 10)                    AS bucket,    -- 1..10
    ROUND(((width_bucket(score, 0.0, 1.0001, 10) - 1) * 0.1)::numeric, 1) AS bucket_min,
    ROUND(( width_bucket(score, 0.0, 1.0001, 10)      * 0.1)::numeric, 1) AS bucket_max,
    COUNT(*)                                                 AS count
FROM detection_events
GROUP BY 1, 2
ORDER BY 1, 2;


-- 3e. Cross-recognizer risk overview: images caught by multiple models at once
--     (one row per batch of concurrent calls from /detect/image, grouped by minute)
--     Useful for finding images that triggered several alarms simultaneously.
CREATE OR REPLACE VIEW high_risk_events AS
SELECT
    date_trunc('minute', created_at)   AS minute,
    COUNT(DISTINCT recognizer)          AS recognizer_hit_count,
    array_agg(DISTINCT recognizer)      AS recognizers_triggered,
    MAX(score)                          AS max_score,
    MIN(score)                          AS min_score
FROM detection_events
WHERE is_flagged = true
GROUP BY 1
HAVING COUNT(DISTINCT recognizer) > 1
ORDER BY 1 DESC;
