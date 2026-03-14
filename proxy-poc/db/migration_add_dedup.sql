-- ============================================================
-- Migration: add image de-duplication columns
-- Run this once against an existing detection_events table.
-- Safe to run multiple times (IF NOT EXISTS / IF NOT EXISTS).
-- ============================================================

-- 1. New columns ---------------------------------------------------------

ALTER TABLE detection_events
    ADD COLUMN IF NOT EXISTS image_hash    text,
    ADD COLUMN IF NOT EXISTS scan_count    int NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS times_flagged int NOT NULL DEFAULT 0;

-- 2. Partial unique index for the upsert ON CONFLICT clause --------------
--    Excludes NULL image_hash rows (legacy records without a hash).

CREATE UNIQUE INDEX IF NOT EXISTS detection_events_image_hash_recognizer_uidx
    ON detection_events (image_hash, recognizer)
    WHERE image_hash IS NOT NULL;
