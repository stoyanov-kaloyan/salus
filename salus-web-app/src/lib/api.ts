const BASE_URL =
    (import.meta.env.VITE_API_URL as string | undefined) ??
    "http://localhost:8000";

async function get<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
): Promise<T> {
    const url = new URL(path, BASE_URL);
    if (params) {
        for (const [k, v] of Object.entries(params)) {
            if (v !== undefined) url.searchParams.set(k, String(v));
        }
    }
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
    return res.json() as Promise<T>;
}

export type RecognizerOverview = {
    recognizer: string;
    total_scans: number;
    flagged_count: number;
    clean_count: number;
    flag_rate_pct: number | null;
    avg_score: number | null;
    avg_flagged_score: number | null;
    avg_clean_score: number | null;
    first_scan_at: string | null;
    last_scan_at: string | null;
};

export type DailyStats = {
    day: string;
    recognizer: string;
    total_scans: number;
    flagged_count: number;
    clean_count: number;
    flag_rate_pct: number | null;
    avg_score: number | null;
    avg_flagged_score: number | null;
};

export type HourlyActivity = {
    hour: string;
    recognizer: string;
    scans: number;
    flagged: number;
};

export type ScoreBucket = {
    recognizer: string;
    bucket: number;
    bucket_min: number;
    bucket_max: number;
    count: number;
};

export type HighRiskEvent = {
    minute: string;
    recognizer_hit_count: number;
    recognizers_triggered: string[];
    max_score: number;
    min_score: number;
};

export type EventRecord = {
    id: string;
    created_at: string;
    recognizer: string;
    endpoint: string;
    is_flagged: boolean;
    label: string;
    score: number;
    image_hash: string | null;
    scan_count: number;
    times_flagged: number;
};

export const api = {
    getSummary: () => get<RecognizerOverview[]>("/stats/summary"),

    getDaily: (params?: { recognizer?: string; days?: number }) =>
        get<DailyStats[]>("/stats/daily", params),

    getRecent: () => get<HourlyActivity[]>("/stats/recent"),

    getDistribution: (recognizer?: string) =>
        get<ScoreBucket[]>(
            "/stats/distribution",
            recognizer ? { recognizer } : undefined,
        ),

    getHighRisk: (limit = 50) =>
        get<HighRiskEvent[]>("/stats/high-risk", { limit }),

    getEvents: (params?: {
        recognizer?: string;
        flagged_only?: boolean;
        limit?: number;
        offset?: number;
    }) => get<EventRecord[]>("/stats/events", params),
};
